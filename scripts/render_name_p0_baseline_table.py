"""Recompute the M0 name baseline from its prediction sidecar and render the table.

Reads benchmarks/results/name-p0-baseline-e5.{json,predictions.jsonl}; no model,
no network, no dataset text. Exits nonzero when any recomputed count differs from
the stored JSON. The sidecar holds spans and text hashes only.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path("benchmarks/results")
LINEAGE = {  # dataset -> (manifest, original-count key, excluded-count key)
    "klue_dev_1000": ("name-span-v1-external-manifest.json", None, None),
    "klue_dev_999": ("name-generalization-final-v1-nonoverlap-heldout.manifest.json",
                     "originally_selected_ids", "excluded_training_or_validation_overlap_ids"),
    "klue_fresh_1000": ("name-multisource-v2-fresh-klue.manifest.json",
                        "original_selected_count", "excluded_count"),
    "kdpii_test_500": ("name-generalization-final-v1-nonoverlap-kdpii.manifest.json", None, None),
    "kdpii_fresh_1992": ("name-multisource-v2-fresh-kdpii-lossless.manifest.json",
                         "original_selected_count", "excluded_count"),
}


def count(value):
    return len(value) if isinstance(value, list) else value


def lineage(name):
    if name == "synthetic_v11_192":
        return 192, 0
    manifest, original_key, excluded_key = LINEAGE[name]
    data = json.loads((RESULTS / manifest).read_text())
    selected = len(data["selected_ids"])
    if original_key is None:
        return selected, 0
    return count(data[original_key]), count(data[excluded_key])


def recompute(path):
    rows = defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        rows[row["dataset"]].append(row)
    out = {}
    for dataset, sentences in rows.items():
        policies = sorted({k[5:-1] for r in sentences for k in r if k.startswith("gold[")})
        for policy in policies:
            tp = fp = fn = full = negative = negative_fp = 0
            for r in sentences:
                pred = {tuple(s) for s in r["predicted"]}
                gold = {tuple(s) for s in r[f"gold[{policy}]"]}
                tp += len(pred & gold)
                fp += len(pred - gold)
                fn += len(gold - pred)
                full += len(r[f"fully_masked[{policy}]"])
                if not gold:
                    negative += 1
                    negative_fp += bool(pred)
            out[(dataset, policy)] = dict(
                sentences=len(sentences), tp=tp, fp=fp, fn=fn, fully_covered_names=full,
                negative_sentences=negative, negative_false_positive_sentences=negative_fp)
    return out


def main():
    stored = json.loads((RESULTS / "name-p0-baseline-e5.json").read_text())
    side = recompute(RESULTS / "name-p0-baseline-e5.predictions.jsonl")
    lines = ["| 자료 | 정책 | 선정 | 중복 제외 | 평가 | 실패 | 정답 | TP/FP/FN | P | R | F1 |"
             " 전체 가림 | 비인물 오탐 문장 |", "|" + "---|" * 13]
    bad = []
    for (dataset, policy), mine in sorted(side.items()):
        spec = stored["datasets"][dataset]["metrics"]
        key = "primary" if "primary" in spec else policy
        overall = spec[key]["overall"]
        for field, value in mine.items():
            if overall[field] != value:
                bad.append((dataset, policy, field, overall[field], value))
        selected, excluded = lineage(dataset)
        gold = mine["tp"] + mine["fn"]
        if gold != overall["gold_names"] or selected - excluded != mine["sentences"]:
            bad.append((dataset, policy, "gold/lineage", overall["gold_names"], gold))
        p = mine["tp"] / (mine["tp"] + mine["fp"])
        r = mine["tp"] / gold
        f1 = 2 * mine["tp"] / (2 * mine["tp"] + mine["fp"] + mine["fn"])
        failed = selected - excluded - mine["sentences"]
        lines.append(
            f"| {dataset} | {policy if key != 'primary' else '-'} | {selected} | {excluded} | "
            f"{mine['sentences']} | {failed} | {gold} | "
            f"{mine['tp']}/{mine['fp']}/{mine['fn']} | "
            f"{p:.2%} | {r:.2%} | {f1:.2%} | {mine['fully_covered_names']} | "
            f"{mine['negative_false_positive_sentences']}/{mine['negative_sentences']} |")
    print("\n".join(lines))
    if bad:
        print("\nMISMATCH:", *bad, sep="\n  ", file=sys.stderr)
        sys.exit(1)
    print("\nsidecar recomputation matches stored JSON for every row")


if __name__ == "__main__":
    main()
