"""GPU re-baseline and nickname-scope contrast inputs for P0 (amendment 2).

One dataset per process, sequential. Settings are fixed before any inference:
float32, autocast off, TF32 off, pinned E5, threshold 0.9. Three passes per text on the
same model: guard-level analyze (current behaviour, unpatched label map), raw unpatched
KoreanNER output, and raw output with personal_handle decoded as a separate experimental
entity. Arm A = raw KR_NAME; arm B = raw KR_NAME + KR_HANDLE, from the same model run.
Only spans, scores and text hashes are written (no text). src/ is not modified.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_name_p0_baseline as base  # noqa: E402

OUT = base.RESULTS / "name-p0-gpu-ab"


def fix_precision(torch):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return dict(dtype="float32", autocast=False, tf32_matmul=False, tf32_cudnn=False,
                matmul_precision=torch.get_float32_matmul_precision())


def spans(results, entity):
    return sorted([r.start, r.end, round(float(r.score), 6)] for r in results
                  if r.entity_type == entity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(base.DATASETS) + ["handle_contexts"],
                        required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch

    import ko_pii_guard.ner as ner_module
    from ko_pii_guard import KoreanPIIGuard
    from ko_pii_guard.ner import MODEL_REVISION, KoreanNER

    precision = fix_precision(torch)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{args.dataset}.jsonl"
    if path.exists():
        parser.error("do not overwrite")
    if args.dataset == "handle_contexts":
        from name_p0_handle_contexts import build, digest

        rows = build()
        cases = [dict(id=r["id"], text=r["text"], meta=r) for r in rows]
        digest_value = digest(rows)
    else:
        cases, _, _ = base.load_cases(args.dataset)
        digest_value = None
    ner = KoreanNER.from_pretrained(device=args.device, score_threshold=0.9)
    guard = KoreanPIIGuard(entities=["KR_NAME"], ner=ner, score_threshold=0.0)
    original_labels = dict(ner_module._LABELS)
    patched_labels = dict(original_labels, personal_handle="KR_HANDLE")
    policies = ({"PS_NAME": None, "PS_NAME+PS_NICKNAME": "alternate_expected"}
                if args.dataset.startswith("kdpii") else {"primary": None})
    started = time.time()
    with path.open("w") as sink:
        for index, case in enumerate(cases, 1):
            text = case["text"]
            guard_pred = sorted([r.start, r.end] for r in guard.analyze(text)
                                if r.entity == "KR_NAME")
            ner_module._LABELS.clear()
            ner_module._LABELS.update(original_labels)
            raw_ref = spans(ner.analyze(text), "KR_NAME")
            ner_module._LABELS.clear()
            ner_module._LABELS.update(patched_labels)
            patched = ner.analyze(text)
            ner_module._LABELS.clear()
            ner_module._LABELS.update(original_labels)
            row = dict(dataset=args.dataset, id=case["id"],
                       text_sha256=base.sha(text), guard_pred=guard_pred,
                       raw_name_unpatched=raw_ref, raw_name=spans(patched, "KR_NAME"),
                       raw_handle=spans(patched, "KR_HANDLE"))
            if "meta" in case:
                row["meta"] = {k: case["meta"][k] for k in ("pair", "kind", "span",
                                                            "expected_protected")}
            else:
                row["source"] = base.source_of(args.dataset, case)
                for policy, field in policies.items():
                    row[f"gold[{policy}]"] = sorted(
                        [e["start"], e["end"]] for e in case[field or "expected"]
                        if e["entity"] == "KR_NAME")
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 200 == 0 or index == len(cases):
                print(args.dataset, index, len(cases), flush=True)
    meta = dict(dataset=args.dataset, rows=len(cases), seconds=round(time.time() - started, 1),
                precision=precision, model_revision=MODEL_REVISION, name_threshold=0.9,
                torch=torch.__version__, cuda=torch.version.cuda, device=args.device,
                gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                python=sys.executable, context_set_sha256=digest_value,
                sidecar_sha256=base.file_sha(path))
    (OUT / f"{args.dataset}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
