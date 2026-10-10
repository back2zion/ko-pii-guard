"""Verify immutable v3 decoder candidates and public historical masking counts."""

import argparse
import hashlib
import json
from pathlib import Path

from name_address_benchmark import _source_metadata, evaluate, load_cases

from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard
from ko_pii_guard.ner import KoreanNER

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/data"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    artifact = ROOT / "artifacts/name-context-v6-filter-retry"
    assert digest(artifact / "name_context.safetensors") == digest(
        ROOT / "artifacts/name-context-v3/name_context.safetensors"
    ), "Baseline name decoder must remain byte-identical"
    sources = _source_metadata()
    hashes = {p.name: digest(p) for p in DATA.glob("*.jsonl")}
    v3 = json.loads((ROOT / "benchmarks/results/name-context-v3-acceptance.json").read_text())
    v3_new = json.loads((ROOT / "artifacts/name-context-v3/training_report.json").read_text())
    v4_new = json.loads((
        ROOT / "benchmarks/results/name-context-v3-on-v4-evaluation.json"
    ).read_text())
    v5_new = json.loads((
        ROOT / "artifacts/name-context-v5-filter/training_report.json"
    ).read_text())
    corpora = {name: load_cases(DATA / name) for name in (
        "name_context.jsonl", "name_address.jsonl", "business_korean.jsonl",
        "name_field_boundaries.jsonl",
    )}
    corpora["v1_retired_evaluation"] = [c for c in load_cases(
        DATA / "name_context_training.jsonl"
    ) if c["split"] == "evaluation"]
    corpora["v2_retired_evaluation"] = load_cases(DATA / "name_context_v2_regressions.jsonl")
    corpora["v3_retired_evaluation"] = load_cases(DATA / "name_context_v3_regressions.jsonl")
    corpora["v4_retired_evaluation"] = load_cases(DATA / "name_context_v4_regressions.jsonl")
    corpora["v5_retired_evaluation"] = [c for c in load_cases(
        DATA / "name_context_v5.jsonl"
    ) if c["split"] == "evaluation"]
    guard = KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=KoreanNER.from_pretrained(
        name_context_path=artifact
    ))
    report = dict(scope="DEVELOPMENT historical contracts; first fresh gate is separate",
                  source=sources, data_sha256=hashes,
                  checkpoint_sha256={p.name: digest(p) for p in artifact.glob("*.safetensors")},
                  results={}, baseline_comparison={})
    passed = True
    for name, cases in corpora.items():
        current = evaluate(cases, guard)
        report["results"][name] = current
        if name == "v3_retired_evaluation":
            before = v3_new["evaluation_candidate"]["counts"]
        elif name == "v4_retired_evaluation":
            before = v4_new["results"]["v4_new_evaluation"]["counts"]
        elif name == "v5_retired_evaluation":
            before = v5_new["evaluation_baseline"]["counts"]
        else:
            before = v3["results"][name]["counts"]
        after = current["counts"]
        ok = (after["true_positive"] == before["true_positive"]
              and after["fully_covered_spans"] >= before["fully_covered_spans"]
              and after["false_positive"] <= before["false_positive"]
              and after["false_positive_sentences"] == 0)
        report["baseline_comparison"][name] = dict(before=before, after=after, passed=ok)
        passed &= ok
        print(name, after, "passed", ok, flush=True)
    report["acceptance_passed"] = passed
    report["source_changed_during_run"] = sources != _source_metadata() or any(
        digest(DATA / name) != expected for name, expected in hashes.items()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    if report["source_changed_during_run"] or (args.check and not passed):
        raise SystemExit("Historical true-name preservation or non-person acceptance failed")


if __name__ == "__main__":
    main()
