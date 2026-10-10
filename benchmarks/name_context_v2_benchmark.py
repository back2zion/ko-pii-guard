"""Acceptance check for known v1 misses and regressions via public APIs.

All these cases have been inspected and used for v2 development. The training
report contains the separate new held-out evaluation. Do not combine the scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from name_address_benchmark import _dependency_versions, _source_metadata, evaluate, load_cases

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
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/name-context-v2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    files = (
        "name_context_training.jsonl",
        "name_context.jsonl",
        "name_address.jsonl",
        "business_korean.jsonl",
        "name_field_boundaries.jsonl",
    )
    hashes = {name: digest(DATA / name) for name in files}
    sources = _source_metadata()
    v1_report_path = ROOT / "artifacts/name-context-v1/training_report.json"
    v1 = json.loads(v1_report_path.read_text())
    missed = set(v1["evaluation_candidate"]["error_ids"]["uncovered_sensitive_span"])
    regressed = set(v1["evaluation_candidate"]["error_ids"]["exact_span"]) - set(
        v1["evaluation_baseline"]["error_ids"]["exact_span"]
    )
    old_evaluation = [
        c for c in load_cases(DATA / "name_context_training.jsonl") if c["split"] == "evaluation"
    ]
    ner = KoreanNER.from_pretrained(name_context_path=args.checkpoint)
    guard = KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=ner)
    report = dict(
        scope="KNOWN DEVELOPMENT acceptance, not independent accuracy",
        measured_at_utc=datetime.now(timezone.utc).isoformat(),
        data_sha256=hashes,
        source=sources,
        dependency_versions=_dependency_versions(),
        benchmark_sha256=digest(Path(__file__)),
        v1_training_report_sha256=digest(v1_report_path),
        checkpoint_sha256={
            name: digest(args.checkpoint / name)
            for name in ("name_context_config.json", "name_context.safetensors")
        },
        known_v1_missed_case_ids=sorted(missed),
        known_v1_regressed_case_ids=sorted(regressed),
        profiles={},
    )
    # Compare v1 and v2 under the same current runtime and public masking path.
    corpora = {"v1_evaluation_now_development": old_evaluation}
    corpora.update({name: load_cases(DATA / name) for name in files[2:]})
    for profile, path in [("v1", ROOT / "artifacts/name-context-v1"), ("v2", args.checkpoint)]:
        ner.name_context_head = load_name_head(
            path, model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu"
        )
        results = {name: evaluate(cases, guard) for name, cases in corpora.items()}
        report["profiles"][profile] = results
        print(profile, {name: r["counts"] for name, r in results.items()}, flush=True)
    # Historical contrast set includes the four known high-confidence errors.
    # Its entire current profile must also stay correct on development data.
    original = evaluate(load_cases(DATA / "name_context.jsonl"), guard)
    report["profiles"]["v2"]["original_contrasts_now_development"] = original
    failed = set(
        report["profiles"]["v2"]["v1_evaluation_now_development"]["error_ids"]["exact_span"]
    )
    report["remaining_known_misses"] = sorted(missed & failed)
    report["remaining_known_regressions"] = sorted(regressed & failed)
    report["acceptance_passed"] = (
        not report["remaining_known_misses"]
        and not report["remaining_known_regressions"]
        and all(
            r["counts"]["false_positive"] == 0 and r["counts"]["false_negative"] == 0
            for r in report["profiles"]["v2"].values()
        )
    )
    report["source_changed_during_run"] = sources != _source_metadata() or any(
        digest(DATA / name) != expected for name, expected in hashes.items()
    )
    if report["source_changed_during_run"]:
        raise RuntimeError("Source or data changed during acceptance check")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("acceptance_passed", report["acceptance_passed"], flush=True)
    if args.check and not report["acceptance_passed"]:
        raise SystemExit("Known name context misses or regressions remain")


if __name__ == "__main__":
    main()
