"""A0/A1 on cached validation features; isolate transition constraints, no training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from name_bioes_ablation import constrained_bioes
from train_name_context import collate
from train_name_span_experiment import metrics

from ko_pii_guard.name_context import (
    HEAD_LABELS,
    add_nonoverlapping_name_results,
    filter_name_results,
    load_name_head,
    rescue_decoders,
)
from ko_pii_guard.ner import MODEL_ID, MODEL_REVISION, _decode_tokens

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/name-span-features-v1.pt"))
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/name-context-v11")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output")
    torch.set_num_threads(2)
    cached = torch.load(args.cache, weights_only=True)
    data = ROOT / "benchmarks/data/name_context_v11.jsonl"
    cases = [json.loads(line) for line in data.read_text().splitlines()]
    cases = [c for c in cases if c["split"] == "validation"]
    if (
        cached["key"]["validation_ids"] != [c["id"] for c in cases]
        or cached["key"]["data_sha256"] != hashlib.sha256(data.read_bytes()).hexdigest()
    ):
        raise ValueError("Feature cache data differs")
    features = cached["validation"]
    del cached
    head = load_name_head(args.baseline, model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu")
    predictions = {"A0_argmax": [], "A1_constrained_BIOES": []}
    with torch.inference_mode():
        for base in range(0, len(cases), 32):
            rows = features[base : base + 32]
            f, c, p, lengths, _ = collate(rows)
            heads = [(head, 0.9), *rescue_decoders(head)]
            outputs = [h(f, c, p, lengths) for h, _ in heads]
            for row, case in enumerate(cases[base : base + 32]):
                for method in predictions:
                    results = []
                    for index, (logits, (_, threshold)) in enumerate(
                        zip(outputs, heads, strict=True)
                    ):
                        values = logits[row, : len(case["text"])]
                        labels = (
                            constrained_bioes(values)
                            if method == "A1_constrained_BIOES"
                            else values.argmax(-1)
                        )
                        scores = values.softmax(-1).gather(1, labels[:, None]).squeeze(1)
                        tokens = {
                            (i, i + 1): (0, HEAD_LABELS[int(label)], float(score))
                            for i, (label, score) in enumerate(zip(labels, scores, strict=True))
                        }
                        candidate = filter_name_results(
                            _decode_tokens(tokens, threshold), rows[row][0], head.span_filter
                        )
                        results = (
                            candidate
                            if index == 0
                            else add_nonoverlapping_name_results(results, candidate)
                        )
                    predictions[method].append({(r.start, r.end) for r in results})
    report = dict(
        scope="cached_synthetic_validation_ablation_not_public_API",
        factor="only legal BIOES transitions; same heads, confidences, thresholds and filter",
        address_limitation="Cached name-only ablation; no backbone address conflict handling",
        results={key: metrics(cases, value) for key, value in predictions.items()},
        sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                ROOT / "scripts/name_bioes_ablation.py",
                data,
                args.cache,
                *args.baseline.glob("*.safetensors"),
            )
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        {
            k: {key: value for key, value in v.items() if key != "errors"}
            for k, v in report["results"].items()
        }
    )


if __name__ == "__main__":
    main()
