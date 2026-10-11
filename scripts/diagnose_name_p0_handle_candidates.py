"""Split remaining nickname misses/false spans by cause using the saved GPU A/B spans (no model).

Stored outputs are thresholded (>= 0.9), so "no output" cannot distinguish a candidate below the
threshold from no candidate at all. Reports only what the saved data can support.
"""

from __future__ import annotations

import json
from pathlib import Path

AB = Path("benchmarks/results/name-p0-gpu-ab")
POLICY = "PS_NAME+PS_NICKNAME"


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def main():
    report = {}
    for name in ("kdpii_test_500", "kdpii_fresh_1992"):
        rows = [json.loads(line) for line in (AB / f"{name}.jsonl").read_text().splitlines()]
        out = dict(gold=0, exact_in_A=0, exact_in_B_only=0, boundary_mismatch_B=0,
                   no_output_B=0, B_false_spans=0, B_false_overlapping_gold=0,
                   B_false_no_gold_overlap=0, B_false_inside_real_name_gold=0,
                   nickname_only_gold=0, nickname_only_exact_in_B=0, nickname_only_overlap_in_B=0)
        for row in rows:
            gold = [tuple(g) for g in row[f"gold[{POLICY}]"]]
            a = [tuple(s[:2]) for s in row["raw_name"]]
            handle = [tuple(s[:2]) for s in row["raw_handle"]]
            b = sorted(set(a) | set(handle))
            real = [tuple(g) for g in row["gold[PS_NAME]"]]
            for g in gold:
                if g not in real:
                    out["nickname_only_gold"] += 1
                    out["nickname_only_exact_in_B"] += g in b
                    out["nickname_only_overlap_in_B"] += any(overlap(g, p) for p in b)
                out["gold"] += 1
                if g in a:
                    out["exact_in_A"] += 1
                elif g in handle:
                    out["exact_in_B_only"] += 1
                elif any(overlap(g, p) for p in b):
                    out["boundary_mismatch_B"] += 1
                else:
                    out["no_output_B"] += 1
            for p in b:
                if p in gold:
                    continue
                out["B_false_spans"] += 1
                if any(overlap(p, g) for g in gold):
                    out["B_false_overlapping_gold"] += 1
                else:
                    out["B_false_no_gold_overlap"] += 1
                if any(overlap(p, g) for g in real):
                    out["B_false_inside_real_name_gold"] += 1
        out["candidate_set_exact_coverage_B"] = round(
            (out["exact_in_A"] + out["exact_in_B_only"]) / out["gold"], 4)
        out["candidate_set_overlap_coverage_B"] = round(
            (out["exact_in_A"] + out["exact_in_B_only"] + out["boundary_mismatch_B"])
            / out["gold"], 4)
        report[name] = out
    report["limit"] = ("Saved spans are thresholded at 0.9; below-threshold candidates were "
                       "not stored, so 'no_output_B' mixes below-threshold candidates and "
                       "absent ones.")
    Path("benchmarks/results/name-p0-handle-candidate-diagnosis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
