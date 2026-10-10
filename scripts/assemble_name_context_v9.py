"""Keep v7 rescue weights and add v8's improvements without replacing the old decoder."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from name_address_benchmark import evaluate, load_cases  # noqa: E402
from name_context_v6_benchmark import RecordedGuard, nonregression_gate  # noqa: E402

from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard  # noqa: E402
from ko_pii_guard.name_context import load_name_head  # noqa: E402
from ko_pii_guard.ner import MODEL_ID, MODEL_REVISION, KoreanNER  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new or empty")
    torch.set_num_threads(2)
    baseline = ROOT / "artifacts/name-context-v7"
    refined = ROOT / "artifacts/name-context-v8"
    assert digest(baseline / "name_context.safetensors") == digest(
        refined / "name_context.safetensors"
    )
    sources = [Path(__file__), ROOT / "benchmarks/name_context_v9_cases.py",
               ROOT / "benchmarks/name_context_v6_benchmark.py",
               ROOT / "benchmarks/name_address_benchmark.py",
               *sorted((ROOT / "src/ko_pii_guard").glob("*.py"))]
    inputs = [*sources, ROOT / "benchmarks/data/name_context_v9.jsonl",
              *[ROOT / f"benchmarks/data/name_context_v{v}_regressions.jsonl" for v in (6, 7, 8)],
              *[p for parent in (baseline, refined) for p in parent.iterdir()
                if p.suffix == ".safetensors" or p.name in (
                    "name_context_config.json", "training_manifest.json", "training_report.json"
                )]]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(input_sha256=hashes,
                    policy="v7 primary and rescue bytes preserved; v8 rescue added after v7; "
                           "v8 learned non-person filter; no further training or threshold tuning",
                    fresh_gate="all v6/v7 public errors zero; preserve v7-correct spans on v8 dev; "
                               "then first v9 evaluation with no lost correct or new false span")
    (args.output / "assembly_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    for name, parent in (("name_context.safetensors", baseline),
                         ("rescue_name_context.safetensors", baseline),
                         ("span_filter.safetensors", refined)):
        shutil.copyfile(parent / name, args.output / name)
    shutil.copyfile(refined / "rescue_name_context.safetensors",
                    args.output / "rescue_name_context_1.safetensors")
    config = json.loads((baseline / "name_context_config.json").read_text())
    refined_config = json.loads((refined / "name_context_config.json").read_text())
    config["span_filter"] = refined_config["span_filter"]
    config["extra_rescue_heads"] = [refined_config["rescue_head"]]
    (args.output / "name_context_config.json").write_text(json.dumps(config, indent=2)+"\n")
    output_hashes = {p.name: digest(p) for p in args.output.iterdir()
                     if p.suffix == ".safetensors" or p.name == "name_context_config.json"}
    report = dict(manifest=manifest, checkpoint_sha256=output_hashes, evaluation_run=False)
    ner = KoreanNER.from_pretrained(name_context_path=args.output)
    known = [c for v in (6, 7) for c in load_cases(
        ROOT / f"benchmarks/data/name_context_v{v}_regressions.jsonl"
    )]
    report["development_candidate"] = evaluate(
        known, KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=ner)
    )
    counts = report["development_candidate"]["counts"]
    dev_ok = (counts["true_positive"] == counts["fully_covered_spans"] == 468
              and counts["false_positive"] == counts["false_negative"] == 0)

    def compare(cases):
        metrics, predictions = {}, {}
        for profile, path in (("baseline", baseline), ("candidate", args.output)):
            ner.name_context_head = load_name_head(
                path, model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu"
            )
            guard = RecordedGuard(KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=ner))
            metrics[profile] = evaluate(cases, guard)
            predictions[profile] = guard.predictions
        return dict(metrics=metrics, span_gate=nonregression_gate(
            cases, predictions["baseline"], predictions["candidate"]
        ))

    report["retired_v8_comparison"] = compare(load_cases(
        ROOT / "benchmarks/data/name_context_v8_regressions.jsonl"
    ))
    dev_ok &= report["retired_v8_comparison"]["span_gate"]["nonregression_passed"]
    if dev_ok:
        report["new_evaluation"] = compare([c for c in load_cases(
            ROOT / "benchmarks/data/name_context_v9.jsonl"
        ) if c["split"] == "evaluation"])
        report["evaluation_run"] = True
    report["source_changed_during_run"] = any(
        digest(ROOT / name) != value for name, value in hashes.items()
    ) or any(digest(args.output / name) != value for name, value in output_hashes.items())
    report["accepted"] = (dev_ok and report["new_evaluation"]["span_gate"]["nonregression_passed"]
                          and not report["source_changed_during_run"])
    (args.output / "assembly_report.json").write_text(json.dumps(report, ensure_ascii=False,
                                                               indent=2)+"\n")
    if not report["accepted"]:
        raise RuntimeError("Development or fresh per-span gate rejected assembled checkpoint")
    print("Accepted", report["new_evaluation"]["metrics"]["candidate"]["counts"], flush=True)


if __name__ == "__main__":
    main()
