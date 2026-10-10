"""Frozen KDPII evaluation with explicit person-name and nickname policies.

Each model runs public analyze and actual stars mask once per selected record.
Both annotation policies reuse those exact predictions and masks. This source
does not establish unseen-data generalization for the upstream E5 model.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
from pathlib import Path

from evaluate_name_generalization import (
    _score,
    actual_mask_gate,
    digest,
    evaluate_guard,
    load_training_text_hashes,
    paired_intervals,
    training_overlap_ids,
    validate_devices,
    validate_selection,
)

ROOT = Path(__file__).resolve().parents[1]
POLICIES = {"PS_NAME": {"PS_NAME"}, "PS_NAME+PS_NICKNAME": {"PS_NAME", "PS_NICKNAME"}}


def validate_cases(cases, selected_ids):
    if [case["id"] for case in cases] != selected_ids or len(set(selected_ids)) != len(cases):
        raise ValueError("Exported cases must match every predeclared selected IDs in order")
    for case in cases:
        for policy, labels in POLICIES.items():
            expected = sorted((a["start"], a["end"]) for a in case["source_annotations"]
                              if a["label"] in labels)
            field = "expected" if policy == "PS_NAME" else "alternate_expected"
            actual = sorted((e["start"], e["end"]) for e in case[field]
                            if e["entity"] == "KR_NAME")
            if expected != actual:
                raise ValueError(f"{policy} gold differs from preserved source annotations")
            if any(not 0 <= start < end <= len(case["text"]) for start, end in actual):
                raise ValueError("Gold offsets must use original sentence coordinates")


def policy_cases(cases, policy):
    if policy == "PS_NAME":
        return cases
    return [dict(case, expected=case["alternate_expected"]) for case in cases]


def evaluate_policies(cases, guard, *, progress=None):
    primary, predictions, masks = evaluate_guard(cases, guard, progress=progress)
    # All records have one source; record-ID suffixes are not separate genres.
    primary["by_source"] = {"kdpii_v1_test": {k: v for k, v in primary.items()
                                            if k not in ("by_source", "errors")}}
    alternate = _score(policy_cases(cases, "PS_NAME+PS_NICKNAME"), predictions, masks)
    alternate["by_source"] = {"kdpii_v1_test": {k: v for k, v in alternate.items()
                                              if k != "errors"}}
    return {"PS_NAME": primary, "PS_NAME+PS_NICKNAME": alternate}, predictions, masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/ko-pii-kdpii-v1"))
    parser.add_argument("--candidate-factory", default="name_generalization_adapter:build_ner")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        parser.error("Use new output paths; frozen evaluations cannot be overwritten")
    selection = json.loads(args.selection.read_text())
    validate_selection(selection, heldout=True)
    source = json.loads((args.source_dir / "source.json").read_text())
    for filename, details in source["files"].items():
        if digest(args.source_dir / filename) != details["sha256"]:
            raise ValueError(f"Frozen KDPII source changed: {filename}")
    sample = json.loads((args.source_dir / "selection.json").read_text())
    module_name, factory_name = args.candidate_factory.rsplit(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    paths = [Path(__file__), Path(module.__file__), args.selection,
             args.source_dir / "source.json", args.source_dir / "validation.json",
             *[args.source_dir / filename for filename in source["files"]],
             *[ROOT / "scripts" / filename for filename in (
                 "evaluate_name_generalization.py", "evaluate_name_span_external.py",
                 "train_name_span_experiment.py", "fetch_kdpii_evaluation.py")],
             *sorted((ROOT / "src/ko_pii_guard").glob("*.py"))]
    hashes = {**selection["frozen_sha256"], **{str(p.resolve()): digest(p) for p in paths}}
    report = dict(
        scope="second_source_KDPII_frozen_candidate", source=source, selection=selection,
        selected_ids=sample["selected_ids"], frozen_sha256=hashes,
        candidate_factory=args.candidate_factory, guard_score_threshold=0.0,
        baseline_ner_score_threshold=0.9, runtime_promotion=False,
        policies=source["policy"],
        pipeline="Actual public analyze and stars mask; both policies share predictions/masks",
        limitations=source["limitations"]
        + f" Only {source['primary_gold_names']} primary names and "
        f"{source['alternate_gold_names_and_nicknames']} including nicknames in this sample: "
        "recall uncertainty is large. Sentence bootstrap may "
        "understate dependence; this evaluation cannot establish zero-error reliability.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    case_lines = (args.source_dir / "cases.jsonl").read_text().splitlines()
    cases = [json.loads(line) for line in case_lines]
    validate_cases(cases, sample["selected_ids"])
    overlap = training_overlap_ids(cases, load_training_text_hashes(selection, heldout=True))
    report["exact_training_or_validation_text_overlap_ids"] = overlap
    if overlap:
        report["evaluation_blocked"] = "training_or_validation_text_overlap; no resampling"
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        raise SystemExit(f"KDPII text overlap detected for IDs {overlap}; no inference")
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
    baseline_guard = KoreanPIIGuard(entities=["KR_NAME"], ner=baseline_ner, score_threshold=0.0)
    old_scores, baseline, baseline_masks = evaluate_policies(
        cases, baseline_guard, progress=lambda i, n: print(f"baseline {i}/{n}", flush=True))
    report["baseline"] = dict(model_id=MODEL_ID, revision=MODEL_REVISION, metrics=old_scores)
    candidate_guard = KoreanPIIGuard(entities=["KR_NAME"], ner=candidate_ner, score_threshold=0.0)
    new_scores, predictions, masks = evaluate_policies(
        cases, candidate_guard, progress=lambda i, n: print(f"candidate {i}/{n}", flush=True))
    report["candidate"] = dict(metrics=new_scores, comparisons={})
    for policy in POLICIES:
        rows = policy_cases(cases, policy)
        report["candidate"]["comparisons"][policy] = dict(
            regression_gate=actual_mask_gate(rows, baseline, predictions, baseline_masks, masks),
            bootstrap=paired_intervals(rows, baseline, predictions, baseline_masks, masks),
        )
    report["source_changed_during_run"] = any(digest(p) != h for p, h in hashes.items())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: {policy: {k: v for k, v in score.items()
                                    if k not in ("errors", "by_source")}
                            for policy, score in report[name]["metrics"].items()}
                      for name in ("baseline", "candidate")}, ensure_ascii=False), flush=True)
    if report["source_changed_during_run"]:
        raise SystemExit("Frozen inputs changed during KDPII evaluation; cannot accept results")


if __name__ == "__main__":
    main()
