"""Amend a pre-inference blocked held-out sample by exact contamination exclusion.

The original IDs and candidate are fixed. Only exact training/validation text
matches are removed; no replacements or resampling are permitted. This changes
the evaluation population, and the amendment is recorded before inference.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
from pathlib import Path

from evaluate_name_generalization import (
    actual_mask_gate,
    collect_ids,
    digest,
    evaluate_guard,
    load_training_text_hashes,
    paired_intervals,
    read_selected_cases,
    training_overlap_ids,
    validate_devices,
    validate_selection,
    validate_split,
)

ROOT = Path(__file__).resolve().parents[1]


def exclude_exact_training_overlap(cases, hashes):
    if hashes is None or not isinstance(hashes, set) or not hashes:
        raise ValueError("Explicit nonempty frozen training hashes are required")
    excluded = training_overlap_ids(cases, hashes)
    retained = [case for case in cases if case["id"] not in set(excluded)]
    if not retained:
        raise ValueError("Contamination exclusion would leave an empty evaluation sample")
    return retained, excluded


def validate_blocked_report(blocked, selection, factory, original_ids):
    if (blocked.get("scope") != "heldout"
            or blocked.get("source_changed_during_run") is not False
            or blocked.get("evaluation_blocked") != "heldout_training_or_validation_text_overlap"
            or not blocked.get("exact_training_or_validation_text_overlap_ids")
            or "baseline" in blocked or "candidate" in blocked):
        raise ValueError("Require an unchanged contamination block recorded before inference")
    if blocked.get("selected_ids") != original_ids:
        raise ValueError("Amendment must retain the original selected IDs without resampling")
    if (blocked.get("selection", {}).get("candidate") != selection["candidate"]
            or blocked.get("candidate_factory") != factory):
        raise ValueError("Amendment requires the same candidate, decoder, threshold, and factory")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/ko-pii-klue-ner"))
    parser.add_argument("--previous-evaluation", type=Path,
                        default=ROOT / "benchmarks/results/name-span-v1-external.json")
    parser.add_argument("--blocked-report", type=Path,
                        default=ROOT / "benchmarks/results"
                        / "name-generalization-final-v1-heldout.json")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--candidate-factory", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        parser.error("Use new paths; preserve original blocked and amended evaluation evidence")
    selection = json.loads(args.selection.read_text())
    validate_selection(selection, heldout=True)
    source_path = args.source_dir / "klue-ner-v1.1_dev.tsv"
    source = json.loads((args.source_dir / "source.json").read_text())
    for filename, metadata in source["files"].items():
        if digest(args.source_dir / filename) != metadata["sha256"]:
            raise ValueError("Official external source changed")
    split = json.loads(args.split_manifest.read_text())
    if (split["source_sha256"] != digest(source_path)
            or split["previous_evaluation_sha256"] != digest(args.previous_evaluation)):
        raise ValueError("Original split sources changed")
    previous = json.loads(args.previous_evaluation.read_text())
    validate_split(split, collect_ids(source_path), previous["selected_ids"])
    original_ids = split["heldout_ids"]
    blocked = json.loads(args.blocked_report.read_text())
    validate_blocked_report(blocked, selection, args.candidate_factory, original_ids)
    module_name, factory_name = args.candidate_factory.rsplit(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    paths = [Path(__file__), Path(module.__file__), args.selection, args.split_manifest,
             args.blocked_report, args.previous_evaluation, args.source_dir / "source.json",
             *[args.source_dir / name for name in source["files"]],
             *[ROOT / "scripts" / name for name in (
                 "evaluate_name_generalization.py", "evaluate_name_span_external.py",
                 "train_name_span_experiment.py")],
             *sorted((ROOT / "src/ko_pii_guard").glob("*.py"))]
    frozen = {**selection["frozen_sha256"], **{str(path.resolve()): digest(path) for path in paths}}
    # Candidate/source/ID checks complete before reading any selected annotations.
    cases = read_selected_cases(source_path, original_ids)
    training_hashes = load_training_text_hashes(selection, heldout=True)
    cases, excluded = exclude_exact_training_overlap(cases, training_hashes)
    if excluded != blocked["exact_training_or_validation_text_overlap_ids"]:
        raise ValueError("Excluded IDs differ from the original pre-inference contamination block")
    report = dict(
        scope="heldout", originally_selected_ids=original_ids,
        selected_ids=[case["id"] for case in cases],
        original_sample_rows=len(original_ids), sample_rows=len(cases),
        excluded_training_or_validation_overlap_ids=excluded,
        exact_training_or_validation_text_overlap_ids=training_overlap_ids(cases, training_hashes),
        selection=selection, candidate_factory=args.candidate_factory,
        frozen_sha256=frozen, source=source, guard_score_threshold=0.,
        ner_baseline_score_threshold=0.9, runtime_promotion=False,
        selection_amendment=dict(
            reason="Original held-out evaluation blocked before inference on exact text overlap",
            filter="Remove exact SHA256 UTF-8 training/validation text matches only",
            model_selection_changed=False, replacements_or_resampling=False,
            original_blocked_report=str(args.blocked_report),
            original_blocked_report_sha256=digest(args.blocked_report),
            recorded_before_inference=True,
        ),
        pipeline="Public analyze and separate actual stars mask on every retained sentence",
        limitations="Amended nonoverlapping subset of the original fixed sample, not the original "
        "1000-row benchmark. Exclusion uses exact text contamination only, never labels, model "
        "predictions or results. Upstream E5 exposure remains unknown. KLUE includes public and "
        "fictional people; sentence bootstrap does not account for source-document dependence.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Recorded amendment before inference: {len(original_ids)} original, "
          f"{len(excluded)} contamination exclusions, {len(cases)} retained", flush=True)
    import torch
    import transformers

    from ko_pii_guard import KoreanPIIGuard
    from ko_pii_guard.ner import MODEL_ID, MODEL_REVISION, KoreanNER

    torch.set_num_threads(2)
    baseline_ner = KoreanNER.from_pretrained(device=args.device)
    candidate_ner = factory(selection["candidate"])
    report["runtime"] = dict(python=platform.python_version(), torch=torch.__version__,
                             transformers=transformers.__version__,
                             devices=validate_devices(baseline_ner, candidate_ner))
    baseline_guard = KoreanPIIGuard(entities=["KR_NAME"], ner=baseline_ner, score_threshold=0.)
    baseline_score, baseline, baseline_masks = evaluate_guard(
        cases, baseline_guard, progress=lambda i, n: print(f"baseline {i}/{n}", flush=True))
    report["baseline"] = dict(model_id=MODEL_ID, revision=MODEL_REVISION, metrics=baseline_score)
    candidate_guard = KoreanPIIGuard(entities=["KR_NAME"], ner=candidate_ner, score_threshold=0.)
    score, predictions, masks = evaluate_guard(
        cases, candidate_guard, progress=lambda i, n: print(f"candidate {i}/{n}", flush=True))
    report["candidate"] = dict(
        metrics=score,
        regression_gate=actual_mask_gate(cases, baseline, predictions, baseline_masks, masks),
        bootstrap=paired_intervals(cases, baseline, predictions, baseline_masks, masks),
    )
    report["source_changed_during_run"] = any(digest(path) != sha for path, sha in frozen.items())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: {key: value for key, value in report[name]["metrics"].items()
                            if key not in ("errors", "by_source")}
                      for name in ("baseline", "candidate")}, ensure_ascii=False), flush=True)
    if report["source_changed_during_run"]:
        raise SystemExit("Frozen inputs changed during amended evaluation; cannot accept results")


if __name__ == "__main__":
    main()
