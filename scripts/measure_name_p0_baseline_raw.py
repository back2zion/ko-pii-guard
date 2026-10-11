"""Secondary P0-0 measurement: raw KoreanNER.analyze output (no guard post-processing, no mask).

Kept separate from the guard-level baseline so raw and post-processed predictions are
never mixed. Same cases, model, threshold 0.9 and device as measure_name_p0_baseline.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_name_p0_baseline as base  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(base.DATASETS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("do not overwrite")
    import torch
    from train_name_span_experiment import gold, metrics

    from ko_pii_guard.ner import KoreanNER

    torch.set_num_threads(2)
    cases, ids, _ = base.load_cases(args.dataset)
    ner = KoreanNER.from_pretrained(device="cpu", score_threshold=0.9)
    entities = set()
    preds = []
    started = time.time()
    for i, case in enumerate(cases, 1):
        results = ner.analyze(case["text"])
        entities.update(r.entity_type for r in results)
        preds.append({(r.start, r.end) for r in results if r.entity_type == "KR_NAME"})
        if i % 100 == 0:
            print(args.dataset, "raw", i, flush=True)
    out = dict(dataset=args.dataset, raw_entities_seen=sorted(entities),
               wall_seconds=round(time.time() - started, 1), by_source={})
    policies = {"PS_NAME": None, "PS_NAME+PS_NICKNAME": "alternate_expected"} \
        if args.dataset.startswith("kdpii") else {"primary": None}
    for policy, field in policies.items():
        out["by_source"][policy] = {}
        for source in sorted({base.source_of(args.dataset, c) for c in cases}) + ["__all__"]:
            idx = [k for k, c in enumerate(cases)
                   if source == "__all__" or base.source_of(args.dataset, c) == source]
            sub = [dict(cases[k], expected=cases[k][field]) if field else cases[k] for k in idx]
            m = metrics(sub, [preds[k] for k in idx])
            m.pop("errors")
            tp, fp, fn = m["tp"], m["fp"], m["fn"]
            m.update(precision=tp / (tp + fp) if tp + fp else 0.0,
                     recall=tp / (tp + fn) if tp + fn else 0.0, sentences=len(sub),
                     gold_names=sum(len(gold(c)) for c in sub))
            out["by_source"][policy][source] = m
    args.output.write_text(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
