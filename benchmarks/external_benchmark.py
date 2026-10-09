"""Frozen, independent K-PII-Bench slice. Download only during evaluation.

Data: Park, Oh and Cha, K-PII-Bench (2026), CC BY 4.0.
https://huggingface.co/datasets/woohyun212/k-pii-bench
No documents are stored in this repository or uploaded by this evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import ko_pii_guard
from ko_pii_guard import KoreanPIIGuard

REVISION = "844f92f7beba98fa964f957ba324017e82451701"
URL = (
    "https://huggingface.co/datasets/woohyun212/k-pii-bench/resolve/"
    f"{REVISION}/data/test_track_a.jsonl"
)
TYPE_MAP = {
    "KR_RRN": "KR_RRN", "KR_PHONE_NUMBER": "PHONE_NUMBER",
    "KR_BANK_ACCOUNT": "KR_ACCOUNT", "KR_DRIVER_LICENSE": "KR_DRIVER_LICENSE",
    "KR_PASSPORT": "KR_PASSPORT", "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "CREDIT_CARD": "CREDIT_CARD",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if args.limit <= 0 or args.offset < 0:
        parser.error("--limit must be positive and --offset nonnegative")
    if not args.cache.exists():
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=60) as response, args.cache.open("wb") as target:
            for i, line in enumerate(response):
                if i < args.offset:
                    continue
                target.write(line)
                if i + 1 == args.offset + args.limit:
                    break
    raw = args.cache.read_bytes()
    records = [json.loads(line) for line in raw.splitlines()[:args.limit]]
    if len(records) != args.limit:
        parser.error("cached slice has too few records; use a fresh cache path")
    guard = KoreanPIIGuard(entities=list(TYPE_MAP.values()))
    counts = {entity: Counter() for entity in TYPE_MAP.values()}
    domains: Counter = Counter()
    skipped: Counter = Counter()
    fully_covered = 0
    mismatches = []
    started = time.perf_counter()
    for record in records:
        text = record["text"]
        domains[record["domain"]] += 1
        gold = set()
        for entity in record["entities"]:
            assert text[entity["start"]:entity["end"]] == entity["surface"]
            if entity["type"] in TYPE_MAP:
                gold.add((TYPE_MAP[entity["type"]], entity["start"], entity["end"]))
            else:
                skipped[entity["type"]] += 1
        findings = guard.analyze(text)
        predicted = {(f.entity, f.start, f.end) for f in findings}
        for kind, values in (("tp", gold & predicted), ("fp", predicted - gold),
                             ("fn", gold - predicted)):
            for entity, _, _ in values:
                counts[entity][kind] += 1
        for _, start, end in gold:
            cursor = start
            for finding in findings:
                if finding.end <= cursor:
                    continue
                if finding.start > cursor:
                    break
                cursor = max(cursor, finding.end)
            fully_covered += cursor >= end
        if predicted != gold:
            mismatches.append({"doc_id": record["doc_id"], "missing": sorted(gold-predicted),
                               "extra": sorted(predicted-gold)})
    elapsed = time.perf_counter() - started
    total = sum(counts.values(), Counter())
    for counter in [*counts.values(), total]:
        for key in ("tp", "fp", "fn"):
            counter.setdefault(key, 0)
    tp, fp, fn = total["tp"], total["fp"], total["fn"]
    report = {
        "label": args.label, "source_url": URL, "revision": REVISION,
        "license": "CC-BY-4.0", "attribution": "Park, Oh and Cha, K-PII-Bench (2026)",
        "sampling": "contiguous records in published test_track_a order",
        "offset": args.offset,
        "records": len(records), "slice_sha256": hashlib.sha256(raw).hexdigest(),
        "python": platform.python_version(), "domains": dict(domains),
        "loaded_source": ko_pii_guard.__file__,
        "coverage_definition": "union of predicted spans irrespective of entity type",
        "type_mapping": TYPE_MAP, "out_of_scope_gold": dict(skipped),
        "per_entity": counts, "total": total,
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        "full_span_coverage": fully_covered / (tp + fn) if tp + fn else 0,
        "seconds": elapsed, "errors": mismatches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "errors"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
