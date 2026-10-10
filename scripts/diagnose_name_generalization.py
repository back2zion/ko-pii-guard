"""Development-only threshold and coverage attribution from cached NER spans.

One model forward per sentence proposes every span on the constrained path.
Threshold replays use the public analyze API with cached, unrounded NER results.
Coverage is a span-union diagnostic, not a separate stars-mask measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import name_generalization_adapter
import torch
from evaluate_name_generalization import read_selected_cases
from name_generalization_adapter import build_ner
from train_name_span_experiment import gate, gold, metrics

from ko_pii_guard import KoreanPIIGuard

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct(cases, score):
    errors = {record["id"]: record for record in score["errors"]}
    predictions = []
    for case in cases:
        error = errors.get(case["id"], {})
        predictions.append((gold(case) - {tuple(s) for s in error.get("missed", [])})
                           | {tuple(s) for s in error.get("false", [])})
    observed = metrics(cases, predictions)
    if any(observed[key] != score[key] for key in ("tp", "fp", "fn")):
        raise ValueError("Baseline reconstruction differs from frozen development score")
    return predictions


def covered(start, end, spans):
    return all(any(a <= i < b for a, b in spans) for i in range(start, end))


def attribute(cases, baseline, candidate, zero_threshold):
    losses = {"confidence_threshold": [], "path_boundary": [], "no_path_proposal": []}
    for case, old, new, zero in zip(cases, baseline, candidate, zero_threshold, strict=True):
        for start, end in sorted(gold(case)):
            if not covered(start, end, old) or covered(start, end, new):
                continue
            reason = ("confidence_threshold" if covered(start, end, zero)
                      else "path_boundary" if any(a < end and start < b for a, b in zero)
                      else "no_path_proposal")
            losses[reason].append((case["id"], start, end))
    return dict(counts={key: len(value) for key, value in losses.items()}, examples=losses)


class CachedNER:
    results = ()

    def analyze(self, text):
        return self.results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path,
                        default=Path("/tmp/ko-pii-klue-ner/klue-ner-v1.1_dev.tsv"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--logit-cache", type=Path,
                        default=Path("/tmp/name-generalization-development-logits.pt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache_path = args.output.with_suffix(".spans.jsonl")
    if args.output.exists() or cache_path.exists() or args.logit_cache.exists():
        parser.error("Use new output paths; do not replace a diagnostic")
    development = json.loads(args.development.read_text())
    if development.get("scope") != "development" or development.get("source_changed_during_run"):
        raise ValueError("Only completed frozen development evaluation may be diagnosed")
    files = [Path(__file__), args.development, args.source,
             args.checkpoint / "config.json", args.checkpoint / "head.safetensors",
             *[ROOT / "scripts" / name for name in (
                 "name_generalization_model.py", "name_bioes_ablation.py",
                 "name_generalization_adapter.py", "evaluate_name_generalization.py",
                 "train_name_span_experiment.py")],
             *sorted((ROOT / "src/ko_pii_guard").glob("*.py"))]
    hashes = {str(path.resolve()): digest(path) for path in files}
    cases = read_selected_cases(args.source, development["selected_ids"])
    baseline = reconstruct(cases, development["baseline"]["metrics"])
    torch.set_num_threads(2)
    ner = build_ner(dict(checkpoint=str(args.checkpoint), device=args.device, threshold=0.0))
    merged_logits = []
    original_decoder = name_generalization_adapter.decode_char_logits

    def record_decoder(logits, threshold=0.9, origin=0):
        merged_logits.append(logits.detach().cpu().clone())
        return original_decoder(logits, threshold=threshold, origin=origin)

    # Diagnostic-only hook: no on-disk adapter mutation and no changed predictions.
    name_generalization_adapter.decode_char_logits = record_decoder
    cached = CachedNER()
    guard = KoreanPIIGuard(entities=["KR_NAME"], ner=cached, score_threshold=0.0)
    predictions = {threshold: [] for threshold in THRESHOLDS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as stream:
        for index, case in enumerate(cases, 1):
            raw = ner.analyze(case["text"])
            stream.write(json.dumps(dict(id=case["id"], spans=[
                dict(entity=r.entity_type, start=r.start, end=r.end, score=r.score)
                for r in raw
            ])) + "\n")
            for threshold in THRESHOLDS:
                cached.results = [r for r in raw if r.entity_type != "KR_NAME"
                                  or r.score >= threshold]
                predictions[threshold].append({(r.start, r.end) for r in guard.analyze(case["text"])
                                               if r.entity == "KR_NAME"})
            if index % 100 == 0 or index == len(cases):
                print(f"cached and replayed {index}/{len(cases)}", flush=True)
    name_generalization_adapter.decode_char_logits = original_decoder
    if len(merged_logits) != len(cases):
        raise ValueError("Expected one original-coordinate character logit tensor per sentence")
    torch.save(dict(selected_ids=[case["id"] for case in cases], logits=merged_logits,
                    frozen_sha256=hashes), args.logit_cache)
    report = dict(
        scope="development_threshold_diagnostic", thresholds=THRESHOLDS,
        selected_ids=[case["id"] for case in cases], source_sha256=hashes,
        baseline={k: v for k, v in metrics(cases, baseline).items() if k != "errors"},
        limitation="Thresholds inspected on development only. Coverage is predicted span union; "
                   "no separate actual mask call. Thresholds cannot repair a different BIOES path.",
        cached_spans=str(cache_path), cached_spans_sha256=digest(cache_path), results={},
        cached_logits=str(args.logit_cache), cached_logits_sha256=digest(args.logit_cache),
    )
    for threshold, predicted in predictions.items():
        score = metrics(cases, predicted)
        report["results"][str(threshold)] = dict(
            metrics=score,
            regression_gate=gate(cases, baseline, predicted),
            baseline_coverage_loss_attribution=attribute(cases, baseline, predicted,
                                                        predictions[0.0]),
        )
        print(threshold, {k: v for k, v in score.items() if k != "errors"},
              report["results"][str(threshold)]["baseline_coverage_loss_attribution"]["counts"],
              flush=True)
    report["source_changed_during_run"] = any(digest(path) != value
                                              for path, value in hashes.items())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["source_changed_during_run"]:
        raise RuntimeError("Frozen diagnostic inputs changed")


if __name__ == "__main__":
    main()
