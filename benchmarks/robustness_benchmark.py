"""Exact-span evaluation on a curated, synthetic regression corpus.

This is a development regression set, not an independent real-world estimate.
Run: uv run python benchmarks/robustness_benchmark.py --output /tmp/robustness.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ko_pii_guard import KoreanPIIGuard

DEFAULT_DATA = Path(__file__).parent / "data" / "robustness.jsonl"


def load_cases(path: Path = DEFAULT_DATA) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def evaluate(cases: list[dict], guard: KoreanPIIGuard) -> dict:
    """Measure exact type/span matches and span coverage independently."""
    counts: Counter = Counter()
    errors = []
    categories: Counter = Counter()
    for case in cases:
        text = case["text"]
        findings = guard.analyze(text)
        expected = {(e["entity"], e["start"], e["end"]) for e in case["expected"]}
        predicted = {(f.entity, f.start, f.end) for f in findings}
        counts["sentences"] += 1
        categories[case["category"]] += 1
        counts["true_positive"] += len(expected & predicted)
        counts["false_positive"] += len(predicted - expected)
        counts["false_negative"] += len(expected - predicted)
        counts["sensitive_spans"] += len(expected)
        if not expected:
            counts["negative_sentences"] += 1
            counts["false_positive_sentences"] += bool(predicted)
        # A wrong entity or clipped match can pass a loose detection check. Check
        # the actual masking output separately, regardless of predicted type.
        masked = guard.mask(text, style="stars")
        missing_coverage = []
        for _, start, end in expected:
            fully_covered = len(masked) == len(text) and masked[start:end] == "*" * (end - start)
            counts["fully_covered_spans"] += fully_covered
            if not fully_covered:
                missing_coverage.append({"start": start, "end": end})
        if expected != predicted or missing_coverage:
            errors.append({
                "id": case["id"], "category": case["category"], "text": text,
                "expected": case["expected"],
                "masked": masked, "uncovered_spans": missing_coverage,
                "predicted": [
                    {"entity": f.entity, "start": f.start, "end": f.end, "text": f.text}
                    for f in findings
                ],
            })
    tp = counts["true_positive"]
    precision = tp / (tp + counts["false_positive"]) if tp + counts["false_positive"] else 0.0
    recall = tp / (tp + counts["false_negative"]) if tp + counts["false_negative"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "corpus": "curated synthetic development regression; not independent evaluation",
        "counts": dict(counts), "categories": dict(categories),
        "exact_span_precision": precision, "exact_span_recall": recall,
        "exact_span_f1": f1,
        "full_sensitive_span_coverage": (
            counts["fully_covered_spans"] / counts["sensitive_spans"]
            if counts["sensitive_spans"] else 0.0
        ),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="Fail if any regression case fails")
    args = parser.parse_args()
    report = evaluate(load_cases(args.data), KoreanPIIGuard())
    report["data_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "errors"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Errors: {len(report['errors'])}; use --output for per-case details.")
    if args.check and report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
