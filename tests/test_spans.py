"""Check interval optimizations against their direct definitions."""

import random

import pytest
from presidio_analyzer import EntityRecognizer, RecognizerResult

from ko_pii_guard.spans import SpanIndex, deduplicate_results, select_non_overlapping

PRIORITY = {"KR_RRN": 0, "PHONE_NUMBER": 1, "KR_ACCOUNT": 2}


def _result(start, end, score=0.5, entity="KR_ACCOUNT"):
    return RecognizerResult(entity, start, end, score)


def _reference_select(results):
    kept = []
    for result in sorted(
        results,
        key=lambda r: (-r.score, PRIORITY.get(r.entity_type, 99), r.start, r.end),
    ):
        if all(result.end <= other.start or result.start >= other.end for other in kept):
            kept.append(result)
    return sorted(kept, key=lambda r: r.start)


@pytest.mark.parametrize(
    "spans, expected",
    [
        ([], []),
        ([_result(0, 10)], [0]),
        ([_result(0, 10), _result(10, 20)], [0, 1]),
        ([_result(0, 10), _result(0, 10)], [0]),
        ([_result(0, 10), _result(2, 8, 0.9)], [1]),
        ([_result(0, 5), _result(3, 8), _result(7, 12)], [0, 2]),
        ([_result(0, 10), _result(0, 10, entity="KR_RRN")], [1]),
        ([_result(0, 10), _result(0, 5)], [1]),
        ([_result(0, 10, 1, "UNKNOWN"), _result(0, 10)], [0]),
    ],
)
def test_greedy_selection_boundaries_and_ties(spans, expected):
    result = select_non_overlapping(spans, PRIORITY)
    assert [id(r) for r in result] == [id(spans[i]) for i in expected]


def test_randomized_greedy_equivalence():
    rng = random.Random(20261009)
    entities = [*PRIORITY, "UNKNOWN"]
    for size in (0, 1, 2, 10, 50, 200):
        for _ in range(30):
            spans = [
                _result(
                    start := rng.randrange(500),
                    start + rng.randrange(1, 50),
                    rng.choice((0.1, 0.4, 0.5, 0.75, 1)),
                    rng.choice(entities),
                )
                for _ in range(size)
            ]
            if spans:
                spans.extend([spans[0], _result(spans[0].start, spans[0].end)])
            before = [(r.start, r.end, r.score, r.entity_type) for r in spans]
            expected = _reference_select(spans)
            actual = select_non_overlapping(spans, PRIORITY)
            assert [id(r) for r in actual] == [id(r) for r in expected]
            assert [(r.start, r.end, r.score, r.entity_type) for r in spans] == before


@pytest.mark.parametrize(
    "start, end, overlaps",
    [(0, 2, False), (2, 3, True), (9, 12, True), (10, 12, False), (20, 21, False)],
)
def test_span_index_boundaries_and_nested_spans(start, end, overlaps):
    index = SpanIndex([_result(2, 10), _result(4, 6), _result(12, 20)])
    assert index.overlaps(start, end) is overlaps


def test_randomized_span_index_equivalence():
    rng = random.Random(20261010)
    for size in (0, 1, 2, 10, 50, 200):
        spans = [
            _result(start := rng.randrange(500), start + rng.randrange(1, 50))
            for _ in range(size)
        ]
        index = SpanIndex(spans)
        for _ in range(500):
            start = rng.randrange(600)
            end = start + rng.randrange(1, 50)
            assert index.overlaps(start, end) == any(
                start < r.end and end > r.start for r in spans
            )


@pytest.mark.parametrize(
    "spans, expected",
    [
        ([], []),
        ([_result(0, 10, 0)], []),
        ([_result(0, 10), _result(0, 10)], [0]),
        ([_result(0, 10), _result(2, 8)], [0]),
        ([_result(0, 10), _result(2, 8, 0.9)], [1, 0]),
        ([_result(0, 5), _result(3, 8)], [0, 1]),
        ([_result(0, 10), _result(2, 8, entity="KR_RRN")], [0, 1]),
        ([_result(0, 5), _result(0, 10)], [1]),
        ([_result(0, 10), _result(10, 20)], [0, 1]),
    ],
)
def test_duplicate_removal_boundaries_and_containment(spans, expected):
    result = deduplicate_results(spans)
    assert [(r.entity_type, r.start, r.end, r.score) for r in result] == [
        (spans[i].entity_type, spans[i].start, spans[i].end, spans[i].score) for i in expected
    ]


def test_randomized_duplicate_removal_matches_presidio():
    rng = random.Random(20261011)
    for size in (0, 1, 2, 10, 50, 200, 1000):
        for _ in range(12):
            spans = [
                _result(
                    start := rng.randrange(500),
                    start + rng.randrange(1, 80),
                    rng.choice((-0.1, 0.0, 0.1, 0.5, 0.75, 1)),
                    rng.choice([*PRIORITY, "UNKNOWN"]),
                )
                for _ in range(size)
            ]
            if spans:
                first = spans[0]
                spans.extend([
                    first,
                    _result(first.start, first.end, first.score, first.entity_type),
                    _result(first.start, first.end, 0.9, first.entity_type),
                ])
            expected = EntityRecognizer.remove_duplicates(spans)
            actual = deduplicate_results(spans)
            assert [(r.entity_type, r.start, r.end, r.score) for r in actual] == [
                (r.entity_type, r.start, r.end, r.score) for r in expected
            ]
