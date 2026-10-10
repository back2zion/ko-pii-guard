"""Collect a newly trained frozen-encoder head on previously consumed IDs only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from evaluate_name_generalization import read_selected_cases
from name_adapted_adapter import build_frozen_ner

from ko_pii_guard.normalization import normalize_text


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def validate_checkpoint(directory):
    """Require completed training and pin weights, configuration, and archives."""
    required = ("report.json", "manifest.json", "config.json", "head.safetensors")
    if any(not (directory / name).is_file() or not (directory / name).stat().st_size
           for name in required):
        raise ValueError("Require a complete checkpoint with nonempty weights and metadata")
    training = json.loads((directory / "report.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    epochs = manifest.get("epochs")
    epoch = training.get("selected_epoch")
    threshold = training.get("threshold")
    if (training.get("source_changed") is not False or type(epochs) is not int or epochs < 1
            or [row.get("epoch") for row in training.get("history", [])]
            != list(range(1, epochs + 1))
            or type(epoch) is not int or not 1 <= epoch <= epochs
            or config.get("selected_epoch") != epoch
            or not isinstance(threshold, (int, float)) or not math.isfinite(threshold)
            or not 0 <= threshold <= 1 or config.get("threshold") != threshold
            or any(not config.get(key) or config.get(key) != manifest.get(key)
                   for key in ("model_id", "revision"))
            or "adaptation" in config):
        raise ValueError("Require a completed frozen-input checkpoint with matching selection")
    source_hashes = manifest.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("Training must pin its source files")
    for path, expected in source_hashes.items():
        if digest(path) != expected:
            raise ValueError(f"Completed training source changed: {path}")
    # The pinned manifest retains training-source provenance. Runtime replay
    # depends on the completed artifact, not on reopening multi-GB feature caches.
    return {str(path.resolve()): digest(path) for path in directory.rglob("*")
            if path.is_file()}


def validate_consumed_source(report, domain, source):
    scope = "development" if domain == "klue" else "second_source_KDPII_frozen_candidate"
    identifiers = report.get("selected_ids")
    if (domain not in ("klue", "kdpii") or report.get("scope") != scope
            or report.get("source_changed_during_run") is not False
            or not report.get("candidate", {}).get("metrics")
            or not report.get("baseline", {}).get("metrics")
            or not isinstance(identifiers, list) or not identifiers
            or any(not isinstance(identifier, str) or not identifier for identifier in identifiers)
            or len(set(identifiers)) != len(identifiers)):
        raise ValueError("Only completed, already consumed reports with unique IDs are allowed")
    hashes = report.get("frozen_sha256", {})
    if hashes.get(str(source.resolve())) != digest(source):
        raise ValueError("Source differs from the completed consumed evaluation")
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError(f"Consumed evaluation input changed: {path}")
    return hashes


def validate_cases(cases, selected_ids):
    if (not cases or [case["id"] for case in cases] != selected_ids
            or len(set(selected_ids)) != len(selected_ids)
            or any(normalize_text(case["text"]).text != case["text"]
                   or not case["text"].strip() for case in cases)):
        raise ValueError("Require exact consumed IDs and identity-normalized coordinates")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--domain", choices=("klue", "kdpii"), required=True)
    parser.add_argument("--consumed-report", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spans_path = args.output.with_suffix(".spans.jsonl")
    manifest_path = args.output.with_suffix(".manifest.json")
    report_path = args.output.with_suffix(".json")
    outputs = (args.output, spans_path, manifest_path, report_path)
    if len({str(path.resolve()) for path in outputs}) != len(outputs):
        parser.error("Cache and companion output paths must be distinct")
    if any(path.resolve().is_relative_to(args.checkpoint.resolve()) for path in outputs):
        parser.error("Cache outputs must be outside the frozen checkpoint directory")
    if any(p.exists() for p in outputs):
        parser.error("Use new outputs and preserve prior experiments")
    hashes = validate_checkpoint(args.checkpoint)
    checkpoint_files = {str(path.resolve()) for path in args.checkpoint.rglob("*")
                        if path.is_file()}
    report = json.loads(args.consumed_report.read_text())
    hashes.update(validate_consumed_source(report, args.domain, args.source))
    cases = (read_selected_cases(args.source, report["selected_ids"])
             if args.domain == "klue" else
             [json.loads(line) for line in args.source.read_text().splitlines()])
    validate_cases(cases, report["selected_ids"])
    files = [Path(__file__), args.consumed_report, args.source,
             *args.checkpoint.glob("*.json"), *args.checkpoint.glob("*.safetensors"),
             *[Path(__file__).with_name(name) for name in (
                 "name_adapted_adapter.py", "name_encoder_adaptation.py",
                 "name_generalization_model.py", "name_coverage_decoding.py",
                 "name_bioes_ablation.py", "evaluate_name_generalization.py",
                 "evaluate_name_span_external.py", "train_name_span_experiment.py")],
             *Path("src/ko_pii_guard").glob("*.py")]
    hashes.update({str(p.resolve()): digest(p) for p in files})
    manifest = dict(scope="consumed_development_cache", domain=args.domain,
                    selected_ids=[c["id"] for c in cases], frozen_sha256=hashes,
                    device="cpu", torch_version=str(torch.__version__),
                    checkpoint=str(args.checkpoint), independent_evaluation=False)
    manifest["training_source_validation"] = (
        "All training manifest source hashes verified before collection; "
        "complete checkpoint including manifest/report/source archives frozen throughout"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    torch.set_num_threads(2)
    ner = build_frozen_ner(dict(checkpoint=str(args.checkpoint), device="cpu",
                               decoder="viterbi", threshold=0.))
    logits, spans = [], []
    ner.on_logits = lambda value: logits.append(value.detach().cpu().clone())
    for i, case in enumerate(cases, 1):
        spans.append([dict(entity=r.entity_type, start=r.start, end=r.end, score=r.score)
                      for r in ner.analyze(case["text"])])
        if i % 100 == 0 or i == len(cases):
            print(f"{args.domain} consumed development {i}/{len(cases)}", flush=True)
    if len(logits) != len(cases) or any(
            value.shape != (len(c["text"]), 5) or not value.is_floating_point()
            or not torch.isfinite(value).all()
            for c, value in zip(cases, logits, strict=True)):
        raise ValueError("Incomplete original-coordinate logits")
    if any(digest(path) != expected for path, expected in hashes.items()):
        raise ValueError("Frozen inference inputs changed")
    if checkpoint_files != {str(path.resolve()) for path in args.checkpoint.rglob("*")
                            if path.is_file()}:
        raise ValueError("Frozen checkpoint files changed during collection")
    torch.save(dict(**manifest, logits=logits, spans=spans,
                    source_changed_during_run=False), args.output)
    spans_path.write_text("".join(json.dumps(dict(id=c["id"], spans=s)) + "\n"
                                 for c, s in zip(cases, spans, strict=True)))
    report_path.write_text(json.dumps(dict(
        scope="consumed_development_cache", source_changed_during_run=False,
        source_sha256={str(p.resolve()): digest(p) for p in (args.output, spans_path)},
        independent_evaluation=False, runtime_promotion=False), indent=2) + "\n")


if __name__ == "__main__":
    main()
