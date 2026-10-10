"""Preserve every v9-correct span and forbid every newly introduced false span."""

import argparse
import hashlib
import json
from pathlib import Path

from name_address_benchmark import _source_metadata, evaluate, load_cases
from name_context_v6_benchmark import RecordedGuard, nonregression_gate

from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard
from ko_pii_guard.name_context import load_name_head
from ko_pii_guard.ner import MODEL_ID, MODEL_REVISION, KoreanNER

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/data"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "evaluation"), required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "artifacts/name-context-v10-rejected")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    paths = {"v9": ROOT / "artifacts/name-context-v9", "candidate": args.checkpoint}
    sources = _source_metadata()
    hashes = {p.name: digest(p) for p in DATA.glob("*.jsonl")}
    checkpoints = {key: {p.name: digest(p) for p in path.iterdir()
                         if p.suffix == ".safetensors" or p.name == "name_context_config.json"}
                   for key, path in paths.items()}
    for name in ("name_context.safetensors", "rescue_name_context.safetensors",
                 "rescue_name_context_1.safetensors"):
        assert checkpoints["v9"][name] == checkpoints["candidate"][name]
    if args.split == "evaluation":
        corpora = {"v10_new_evaluation": [c for c in load_cases(DATA / "name_context_v10.jsonl")
                                         if c["split"] == "evaluation"]}
    else:
        corpora = {name: load_cases(DATA / name) for name in (
            "name_context.jsonl", "name_address.jsonl", "business_korean.jsonl",
            "name_field_boundaries.jsonl", "name_context_v2_regressions.jsonl",
            "name_context_v3_regressions.jsonl", "name_context_v4_regressions.jsonl",
            "name_context_v6_regressions.jsonl", "name_context_v7_regressions.jsonl",
            "name_context_v8_regressions.jsonl", "name_context_v9_regressions.jsonl",
        )}
        for version in (1, 5):
            name = "name_context_training.jsonl" if version == 1 else "name_context_v5.jsonl"
            corpora[f"v{version}_retired_evaluation"] = [c for c in load_cases(DATA / name)
                                                       if c["split"] == "evaluation"]
    report = dict(scope=args.split, source=sources, data_sha256=hashes,
                  checkpoint_sha256=checkpoints, benchmark_sha256=digest(Path(__file__)),
                  results={})
    ner = KoreanNER.from_pretrained()
    passed = True
    for name, cases in corpora.items():
        metrics, predictions = {}, {}
        for profile, path in paths.items():
            ner.name_context_head = load_name_head(
                path, model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu"
            )
            guard = RecordedGuard(KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=ner))
            metrics[profile] = evaluate(cases, guard)
            predictions[profile] = guard.predictions
        gate = nonregression_gate(cases, predictions["v9"], predictions["candidate"])
        ok = gate["nonregression_passed"]
        if name in ("name_context_v6_regressions.jsonl", "name_context_v7_regressions.jsonl",
                    "name_context_v9_regressions.jsonl"):
            counts = metrics["candidate"]["counts"]
            ok &= (counts["true_positive"] == 234
                   and counts["false_positive"] == counts["false_negative"] == 0)
        report["results"][name] = dict(metrics=metrics, span_gate=gate, passed=ok)
        passed &= ok
        print(name, metrics["candidate"]["counts"], "gate", gate, "passed", ok, flush=True)
    report["acceptance_passed"] = passed
    report["source_changed_during_run"] = (
        digest(Path(__file__)) != report["benchmark_sha256"]
        or sources != _source_metadata()
    ) or any(
        digest(DATA / name) != expected for name, expected in hashes.items()
    ) or any(digest(paths[key] / name) != expected for key, values in checkpoints.items()
             for name, expected in values.items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    if report["source_changed_during_run"] or (args.check and not passed):
        raise SystemExit("Per-span preservation, new false span, or known misses gate failed")


if __name__ == "__main__":
    main()
