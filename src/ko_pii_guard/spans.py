"""Interval operations for non-empty, half-open PII spans.

Keep overlap handling bounded for documents containing many findings. Touching
spans such as [0, 3) and [3, 6) do not overlap.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence

from presidio_analyzer import RecognizerResult


class SpanIndex:
    """Index a fixed collection of spans for O(log n) overlap queries."""

    __slots__ = ("_starts", "_max_ends")

    def __init__(self, results: Sequence[RecognizerResult]):
        spans = sorted((r.start, r.end) for r in results)
        self._starts = [start for start, _ in spans]
        self._max_ends: list[int] = []
        maximum = -1
        for _, end in spans:
            maximum = max(maximum, end)
            self._max_ends.append(maximum)

    def overlaps(self, start: int, end: int) -> bool:
        """Return whether any indexed span overlaps [start, end)."""
        last = bisect_left(self._starts, end) - 1
        return last >= 0 and self._max_ends[last] > start


class _Fenwick:
    """Prefix counts with O(log n) additions and queries (1-based indices)."""

    __slots__ = ("_tree",)

    def __init__(self, size: int):
        self._tree = [0] * (size + 1)

    def add(self, index: int) -> None:
        while index < len(self._tree):
            self._tree[index] += 1
            index += index & -index

    def prefix(self, index: int) -> int:
        count = 0
        while index:
            count += self._tree[index]
            index -= index & -index
        return count


class _PrefixMax:
    """Prefix maximum with monotonic updates at 1-based indices."""

    __slots__ = ("_tree",)

    def __init__(self, size: int):
        self._tree = [-1] * (size + 1)

    def add(self, index: int, value: int) -> None:
        while index < len(self._tree):
            self._tree[index] = max(self._tree[index], value)
            index += index & -index

    def prefix(self, index: int) -> int:
        maximum = -1
        while index:
            maximum = max(maximum, self._tree[index])
            index -= index & -index
        return maximum


def deduplicate_results(results: Sequence[RecognizerResult]) -> list[RecognizerResult]:
    """Presidio-compatible duplicate/contained-span removal in O(n log n).

    A result is discarded if an already-kept result of the same entity contains
    it. Higher scores are considered first, then earlier starts and longer spans.
    Crossing spans and differently typed nested spans are retained. Zero-score
    results are removed, matching EntityRecognizer.remove_duplicates.
    """
    ordered = sorted(set(results), key=lambda r: (-r.score, r.start, -(r.end - r.start)))
    starts_by_entity: dict[str, set[int]] = {}
    for result in ordered:
        starts_by_entity.setdefault(result.entity_type, set()).add(result.start)
    indices = {
        entity: {start: index for index, start in enumerate(sorted(starts), start=1)}
        for entity, starts in starts_by_entity.items()
    }
    max_ends = {entity: _PrefixMax(len(starts)) for entity, starts in indices.items()}
    kept = []
    for result in ordered:
        if result.score == 0:
            continue
        index = indices[result.entity_type][result.start]
        tree = max_ends[result.entity_type]
        if tree.prefix(index) >= result.end:
            continue
        kept.append(result)
        tree.add(index, result.end)
    return kept


def select_non_overlapping(
    results: Sequence[RecognizerResult], priority: Mapping[str, int]
) -> list[RecognizerResult]:
    """Greedily retain the highest-scoring spans in O(n log n) time.

    Preserve score, entity priority, start, and end tie-breaking, including the
    stable input ordering of exact ties. Returned objects are not modified.
    """
    if len(results) < 2:
        return list(results)

    coordinates = sorted({point for r in results for point in (r.start, r.end)})
    indices = {point: index for index, point in enumerate(coordinates, start=1)}
    starts = _Fenwick(len(coordinates))
    ends = _Fenwick(len(coordinates))
    kept: list[RecognizerResult] = []
    for result in sorted(
        results,
        key=lambda r: (-r.score, priority.get(r.entity_type, 99), r.start, r.end),
    ):
        # Recognizers frequently emit several candidates for the same value.
        # Reject a local conflict directly before querying the whole index.
        if kept and result.start < kept[-1].end and result.end > kept[-1].start:
            continue
        start, end = indices[result.start], indices[result.end]
        # Of intervals starting before this end, remove those ending at or
        # before this start. What remains overlaps the candidate.
        if starts.prefix(end - 1) > ends.prefix(start):
            continue
        kept.append(result)
        starts.add(start)
        ends.add(end)
    return sorted(kept, key=lambda r: r.start)
