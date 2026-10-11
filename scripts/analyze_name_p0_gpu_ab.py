"""Score the GPU baseline and the A/B nickname-scope contrast from saved spans (no model).

Reads benchmarks/results/name-p0-gpu-ab/*.jsonl and the CPU baseline sidecar. Bootstrap per
amendment 2: paired, 10,000 resamples, two-sided 95% percentile, pooled TP/FP/FN, seed 20261020,
KDPII records grouped by the ID prefix before the first underscore.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path("benchmarks/results")
AB = RESULTS / "name-p0-gpu-ab"
DATASETS = ["klue_dev_1000", "klue_dev_999", "klue_fresh_1000", "kdpii_test_500",
            "kdpii_fresh_1992", "synthetic_v11_192"]
SEED, RESAMPLES = 20261020, 10000


def load(name):
    return [json.loads(line) for line in (AB / f"{name}.jsonl").read_text().splitlines()]


def pairs(spans):
    return {(s[0], s[1]) for s in spans}


def counts(pred, gold):
    return len(pred & gold), len(pred - gold), len(gold - pred)


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def score(values):
    tp, fp, fn = (int(v) for v in values)
    return dict(tp=tp, fp=fp, fn=fn, f1=round(f1(tp, fp, fn), 5))


def group_key(dataset, row):
    return row["id"].split("_")[0] if dataset.startswith("kdpii") else row["id"]


def bootstrap(matrix, groups):
    rng = np.random.default_rng(SEED)
    unique = sorted(set(groups))
    index = {g: i for i, g in enumerate(unique)}
    per_group = np.zeros((len(unique), matrix.shape[1]))
    for row, g in zip(matrix, groups, strict=True):
        per_group[index[g]] += row
    deltas = []
    for _ in range(RESAMPLES):
        total = per_group[rng.integers(0, len(unique), len(unique))].sum(0)
        deltas.append(f1(*total[3:]) - f1(*total[:3]))
    low, high = np.percentile(deltas, [2.5, 97.5])
    return dict(groups=len(unique), delta_f1_ci95=[round(float(low), 5), round(float(high), 5)])


def analyze_dataset(name, cpu):
    rows = load(name)
    out = dict(rows=len(rows))
    out["raw_unpatched_equals_arm_A"] = all(
        r["raw_name"] == r["raw_name_unpatched"] for r in rows)
    out["guard_gpu_vs_cpu_predictions_differing_sentences"] = sum(
        pairs(r["guard_pred"]) != {tuple(s) for s in cpu[(name, r["id"])]} for r in rows)
    policies = sorted(k[5:-1] for k in rows[0] if k.startswith("gold["))
    out["policies"] = {}
    for policy in policies:
        matrix, groups = [], []
        guard_total = np.zeros(3)
        lost = new_fp = added = added_tp = 0
        for r in rows:
            gold = pairs(r[f"gold[{policy}]"])
            a = pairs(r["raw_name"])
            b = a | pairs(r["raw_handle"])
            guard_total += counts(pairs(r["guard_pred"]), gold)
            tp_a, fp_a, fn_a = counts(a, gold)
            tp_b, fp_b, fn_b = counts(b, gold)
            matrix.append([tp_a, fp_a, fn_a, tp_b, fp_b, fn_b])
            groups.append(group_key(name, r))
            lost += len((a & gold) - b)
            new_fp += len((b - a) - gold)
            added += len(b - a)
            added_tp += len((b - a) & gold)
        total = np.array(matrix).sum(0)
        entry = dict(
            guard_gpu_A=score(guard_total),
            raw_A=score(total[:3]),
            raw_B=score(total[3:]),
            gold_names=int(total[0] + total[2]),
            delta_f1=round(f1(*total[3:]) - f1(*total[:3]), 5),
            lost_exact_spans_A_to_B=lost, new_false_positive_spans_in_B=new_fp,
            spans_added_by_B=added, added_spans_matching_gold=added_tp)
        if policy != "primary" and not name.startswith("synthetic"):
            entry["bootstrap"] = bootstrap(np.array(matrix, dtype=float), groups)
        out["policies"][policy] = entry
    if not any(p.startswith("PS_NAME+") for p in policies):
        out["unannotated_handle_spans_added_by_B"] = sum(
            len(pairs(r["raw_handle"]) - pairs(r["raw_name"])) for r in rows)
        out["note"] = ("No nickname gold: added spans are unannotated, "
                       "not scored as false positives.")
    return out


def analyze_contexts():
    rows = load("handle_contexts")
    result = defaultdict(lambda: defaultdict(int))
    newly = []
    for r in rows:
        s, e = r["meta"]["span"]
        def hit(spans, s=s, e=e):
            return any(a < e and s < b for a, b, _ in spans)

        a_flag = hit(r["raw_name"])
        b_flag = a_flag or hit(r["raw_handle"])
        kind = r["meta"]["kind"]
        result[kind]["n"] += 1
        result[kind]["flagged_A"] += a_flag
        result[kind]["flagged_B"] += b_flag
        if b_flag and not a_flag and kind == "non_personal":
            newly.append(r["id"])
    return dict(by_kind={k: dict(v) for k, v in result.items()},
                non_personal_newly_flagged_by_B=len(newly), newly_flagged_ids=newly)


def main():
    cpu = {}
    for line in (RESULTS / "name-p0-baseline-e5.predictions.jsonl").read_text().splitlines():
        r = json.loads(line)
        cpu[(r["dataset"], r["id"])] = r["predicted"]
    report = dict(scope="name_p0_gpu_ab_results", bootstrap=dict(resamples=RESAMPLES, seed=SEED),
                  datasets={n: analyze_dataset(n, cpu) for n in DATASETS},
                  paired_contexts=analyze_contexts(),
                  meta={p.name: json.loads(p.read_text()) for p in sorted(AB.glob("*.meta.json"))})
    (RESULTS / "name-p0-gpu-ab-results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "meta"}, ensure_ascii=False,
                     indent=1)[:6000])


if __name__ == "__main__":
    main()
