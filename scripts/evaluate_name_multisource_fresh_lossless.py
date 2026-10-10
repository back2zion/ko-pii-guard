"""Resume KDPII with lossless identical-annotation compatibility only.

This deliberately copies the frozen original fresh driver to preserve its source
hash while KLUE is running. Its selection, model, policies, IDs and metric helpers
are unchanged. Only repeated identical KDPII annotation instances are supported.
The failed original manifest, new parser and duplicate metadata are pinned before
any inference. No record is dropped or replaced; this never promotes a runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import evaluate_name_multisource_live as live
import kdpii_lossless_annotations as lossless
import name_reserve_text_audit as audit

ORIGINAL_DRIVER = live.ROOT / "scripts/evaluate_name_multisource_fresh.py"
REFERENCE_CONVERTER = live.ROOT / "scripts/fetch_kdpii_evaluation.py"


def validate_parser_continuation(path, identity, amendment_path, population):
    original = live.read_json(path)
    expected = dict(identity, scope="multisource_live_heldout", domain="kdpii",
                    amendment_sha256=live.digest(amendment_path),
                    **{key: population[key] for key in (
                        "selected_ids", "original_selected_ids", "excluded_ids")})
    if any(original.get(key) != value for key, value in expected.items()):
        raise ValueError("Parser continuation differs from the original frozen evaluation")
    live.verify_hashes(original.get("frozen_sha256"))
    if original["frozen_sha256"].get(str(ORIGINAL_DRIVER)) != live.digest(ORIGINAL_DRIVER):
        raise ValueError("Original fresh driver is not pinned by the failed manifest")
    return original


def run(args):
    if args.domain != "kdpii":
        raise ValueError("This compatibility continuation is only for KDPII")
    manifest_path = args.output.with_suffix(".manifest.json")
    if (args.output.resolve() == manifest_path.resolve()
            or args.output.exists() or manifest_path.exists()):
        raise ValueError("Require distinct unused output and manifest paths")
    chosen, development, artifact = live.validate_selected_decision(
        args.decision, args.development, args.checkpoint)
    identity = dict(decision_sha256=live.digest(args.decision),
                    development_report_sha256=live.digest(args.development),
                    artifact_sha256=artifact, selected_policy=chosen["policy"],
                    reserve_sha256=live.digest(args.reserve), device=args.device,
                    runtime_signature=live.runtime_signature())
    # The successful live reports bind to the ORIGINAL reserve, not its amendment.
    live.validate_live_reports([live.read_json(path) for path in args.live_development], identity)
    amendment = audit.validate_amendment(args.amendment, args.reserve)
    population = amendment["domains"][args.domain]
    selected_ids = population["selected_ids"]
    original = validate_parser_continuation(args.original_failure_manifest, identity,
                                            args.amendment, population)
    reserve = live.read_json(args.reserve)
    source = audit.source_paths(reserve)[args.domain]
    exclusion = Path(reserve["exclusion_hashes_path"])
    preparation = live.read_json(args.checkpoint / "data-manifest.json")
    if (preparation.get("input_sha256", {}).get(str(exclusion))
            != reserve.get("exclusion_hashes_sha256")
            or live.digest(exclusion) != reserve.get("exclusion_hashes_sha256")):
        raise ValueError("Reserve text exclusion differs from frozen training preparation")
    consumed = live.CONSUMED_REPORTS[args.domain]
    if development["frozen_sha256"].get(str(consumed.resolve())) != live.digest(consumed):
        raise ValueError("Consumed report is not pinned by development selection")
    exact, normalized, ancestry_hashes = live.load_ancestry(args.checkpoint)
    overlap = dict(exact_ids=[identifier for identifier in selected_ids
                             if population["text_sha256"][identifier]["exact_sha256"] in exact],
                   normalized_ids=[identifier for identifier in selected_ids
                                   if population["text_sha256"][identifier]["normalized_sha256"]
                                   in normalized])
    factory, factory_name = live.resolve_factory(chosen["policy"])
    hashes = {**development["frozen_sha256"], **original["frozen_sha256"],
              **artifact, **ancestry_hashes,
              **amendment["frozen_sha256"], **live.model_snapshot_hashes()}
    paths = [Path(__file__), args.decision, args.development, args.reserve, args.amendment,
             source, consumed, exclusion, args.original_failure_manifest,
             ORIGINAL_DRIVER, REFERENCE_CONVERTER, *args.live_development,
             *sorted((live.ROOT / "src/ko_pii_guard").glob("*.py"))]
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().is_relative_to(live.ROOT / "scripts"):
            paths.append(Path(filename))
    hashes.update({str(path.resolve()): live.digest(path) for path in paths})
    live.verify_hashes(hashes)
    # Original text-only ancestry checks still precede annotation conversion.
    cases, duplicates = (lossless.read_cases_lossless(source, selected_ids)
                         if not any(overlap.values()) else ([], []))
    parser_compatibility = dict(
        version=lossless.VERSION, duplicate_records=duplicates,
        selected_count=len(selected_ids), dropped_records=0, gold_policy_changed=False,
    )
    parser_paths = dict(helper=Path(lossless.__file__), reference_converter=REFERENCE_CONVERTER,
                        original_driver=ORIGINAL_DRIVER, driver=Path(__file__),
                        original_failure_manifest=args.original_failure_manifest)
    for name, path in parser_paths.items():
        parser_compatibility[name + "_path"] = str(path.resolve())
        parser_compatibility[name + "_sha256"] = live.digest(path)
    report = dict(
        scope="multisource_live_heldout", domain=args.domain, selected_id=chosen["id"],
        selected_ids=selected_ids, consumed_report=str(consumed.resolve()), **identity,
        frozen_sha256=hashes, runtime=dict(identity["runtime_signature"]),
        candidate_factory=factory_name, model_kind="frozen", ancestry_overlap=overlap,
        parser_compatibility=parser_compatibility,
        baseline_model=dict(model_id=live.MODEL_ID, revision=live.MODEL_REVISION, threshold=.9),
        guard_score_threshold=0., runtime_promotion=False, resampling=False,
        amendment_path=str(args.amendment.resolve()), amendment_sha256=live.digest(args.amendment),
        amendment_created_at_utc=amendment["created_at_utc"], amendment_reason=amendment["reason"],
        original_selected_ids=population["original_selected_ids"],
        excluded_ids=population["excluded_ids"],
        original_selected_count=population["original_count"],
        excluded_count=population["excluded_count"], selected_count=population["selected_count"],
        limitations="Sentence bootstrap does not establish dialogue independence. "
                    "Upstream E5 training exposure is not proven absent. Text duplicates were "
                    "excluded before opening annotations; no replacement was sampled.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    report["pre_inference_manifest_path"] = str(manifest_path.resolve())
    report["pre_inference_manifest_sha256"] = live.digest(manifest_path)
    hashes[str(manifest_path.resolve())] = report["pre_inference_manifest_sha256"]

    def finish(**values):
        changed = [path for path, expected in hashes.items()
                   if not Path(path).is_file() or live.digest(path) != expected]
        report.update(**values, source_changed_during_run=bool(changed), changed_inputs=changed)
        report["passed"] = report.get("passed", False) and not changed
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    if any(overlap.values()):
        finish(passed=False, evaluation_blocked="ancestry_text_overlap_no_resampling")
        raise SystemExit("Ancestry overlap; no annotation access, inference, or resampling")
    if [row["id"] for row in cases] != selected_ids or len(set(selected_ids)) != len(cases):
        raise ValueError("Cases differ from amended IDs; no replacement permitted")
    if any(audit.text_hashes(row["text"]) != population["text_sha256"][row["id"]]
           for row in cases):
        raise ValueError("Annotation parser text differs from the text-only amendment")
    lossless.validate_cases_lossless(cases, selected_ids)
    # Independent parser verification after text-only ancestry checking.
    if live.ancestry_overlaps(cases, exact, normalized) != overlap:
        raise ValueError("Parsed ancestry differs from the text-only audit")
    live.torch.set_num_threads(2)
    baseline_ner = live.KoreanNER.from_pretrained(device=args.device, score_threshold=.9)
    candidate_ner = factory(dict(checkpoint=str(args.checkpoint), model_kind="frozen",
                                 device=args.device, **chosen["policy"]))
    report["runtime"]["devices"] = live.validate_devices(baseline_ner, candidate_ner)
    guards = [live.TracedGuard(live.KoreanPIIGuard(
        entities=["KR_NAME"], score_threshold=0., ner=ner))
        for ner in (baseline_ner, candidate_ner)]
    outputs = [live.evaluate_guard(cases, guard, progress=lambda i, n, label=label: print(
        f"{label} {i}/{n}", flush=True)) for label, guard in zip(("baseline", "candidate"),
                                                               guards, strict=True)]
    (_, old, old_masks), (_, new, new_masks) = outputs
    sources = live.compare_sources(cases, args.domain, old, new, old_masks, new_masks)
    finish(sources=sources, passed=all(row["passed"] for row in sources.values()))
    print(json.dumps(dict(passed=report["passed"], selected_count=len(cases),
                          sources={name: dict(metrics=live.compact(row["metrics"]),
                                              passed=row["passed"])
                                   for name, row in sources.items()})), flush=True)
    if not report["passed"]:
        raise SystemExit("Fresh validation failed; runtime promotion remains blocked")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("kdpii",), default="kdpii")
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reserve", type=Path,
                        default=live.ROOT / "benchmarks/results/name-multisource-v2-reserve.json")
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--original-failure-manifest", type=Path, required=True)
    parser.add_argument("--live-development", type=Path, nargs=2, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
