"""Reproduce overlap-operation timings; this is not end-to-end throughput.

Run: uv run python benchmarks/performance_benchmark.py --sizes 100 1000 5000
Reports median wall time and checks exact behavioral equivalence first. The
separated scenario models many independent values (the old scans' worst case),
mixed uses overlapping groups, and overlapping makes all spans overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
from collections.abc import Callable
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from presidio_analyzer import EntityRecognizer, RecognizerResult

from ko_pii_guard.spans import SpanIndex, deduplicate_results, select_non_overlapping

PRIORITY = {"KR_RRN": 0, "PHONE_NUMBER": 1, "KR_ACCOUNT": 2}


def _reference_select(results):
    kept = []
    for result in sorted(
        results,
        key=lambda r: (-r.score, PRIORITY.get(r.entity_type, 99), r.start, r.end),
    ):
        if all(result.end <= other.start or result.start >= other.end for other in kept):
            kept.append(result)
    return sorted(kept, key=lambda r: r.start)


def _reference_veto(candidates, protected):
    return [
        r for r in candidates
        if not any(r.start < p.end and r.end > p.start for p in protected)
    ]


def _indexed_veto(candidates, protected):
    index = SpanIndex(protected)
    return [r for r in candidates if not index.overlaps(r.start, r.end)]


def _median_ms(operation: Callable[[], object], repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = perf_counter()
        operation()
        times.append((perf_counter() - start) * 1000)
    return statistics.median(times)


def _make_spans(size: int, scenario: str, rng: random.Random):
    if scenario == "separated":
        intervals = [(i * 30, i * 30 + 10) for i in range(size)]
        protected_intervals = [(i * 30 + 15, i * 30 + 25) for i in range(size)]
    elif scenario == "mixed":
        intervals = [(i // 3 * 40 + i % 3 * 5, i // 3 * 40 + i % 3 * 5 + 20)
                     for i in range(size)]
        protected_intervals = [(i * 80, i * 80 + 20) for i in range(max(1, size // 6))]
    else:
        intervals = [(i, i + size + 1) for i in range(size)]
        protected_intervals = [(0, size * 2 + 1)]
    spans = [
        RecognizerResult("KR_ACCOUNT", start, end, rng.choice((0.4, 0.75)))
        for start, end in intervals
    ]
    rng.shuffle(spans)
    protected = [
        RecognizerResult("PHONE_NUMBER", start, end, 0.5)
        for start, end in protected_intervals
    ]
    return spans, protected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 1000, 5000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--scenarios", nargs="+", choices=["separated", "mixed", "overlapping"],
        default=["separated", "mixed", "overlapping"],
    )
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.repeats < 1:
        parser.error("sizes and repeats must be positive")
    seed = 20261009
    rng = random.Random(seed)
    rows = []
    for scenario, size in ((scenario, size) for scenario in args.scenarios for size in args.sizes):
        spans, protected = _make_spans(size, scenario, rng)
        operations = (
            (
                "presidio_duplicate_removal",
                partial(EntityRecognizer.remove_duplicates, spans),
                partial(deduplicate_results, spans),
            ),
            (
                "greedy_selection",
                partial(_reference_select, spans),
                partial(select_non_overlapping, spans, PRIORITY),
            ),
            (
                "account_overlap_veto",
                partial(_reference_veto, spans, protected),
                partial(_indexed_veto, spans, protected),
            ),
        )
        for name, reference, optimized in operations:
            assert [id(r) for r in reference()] == [id(r) for r in optimized()]
            previous_ms = _median_ms(reference, args.repeats)
            indexed_ms = _median_ms(optimized, args.repeats)
            rows.append({
                "operation": name,
                "scenario": scenario,
                "candidate_count": size,
                "protected_count": len(protected) if name == "account_overlap_veto" else 0,
                "previous_median_ms": round(previous_ms, 4),
                "optimized_median_ms": round(indexed_ms, 4),
                "speedup": round(previous_ms / indexed_ms, 2),
            })
    print(json.dumps({
        "scope": "overlap operations only; synthetic spans; not end-to-end",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference": "installed Presidio dedupe; v0.2.0 greedy/veto copied in this script",
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "spans_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "src/ko_pii_guard/spans.py").read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "presidio_analyzer": version("presidio-analyzer"),
        "seed": seed,
        "repeats": args.repeats,
        "results": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
