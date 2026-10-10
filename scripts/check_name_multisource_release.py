"""Check completed multisource evaluations for one already selected candidate.

This command loads reports and hashes frozen files. An optional amendment is
recomputed using its text-only auditor. Evaluation annotations and models are
never loaded, policies are never searched, and runtimes are never promoted. Passing is an
external quality gate, not a claim about historical contracts or perfect safety.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path

from evaluate_name_multisource_live import (
    DOMAIN_SOURCES,
    MODEL_ID,
    MODEL_REVISION,
    REQUIRED_SOURCES,
    ROOT,
    compact,
    digest,
    read_json,
    strict_gate_passes,
    validate_live_reports,
    validate_selected_decision,
    verify_hashes,
)
from name_multisource_selection import HIGHER, LOWER, metric_problem, select_candidates

FAILURES = (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError)
FRESH_DRIVER = ROOT / "scripts/evaluate_name_multisource_fresh.py"
LOSSLESS_FRESH_DRIVER = ROOT / "scripts/evaluate_name_multisource_fresh_lossless.py"
LOSSLESS_HELPER = ROOT / "scripts/kdpii_lossless_annotations.py"
REFERENCE_CONVERTER = ROOT / "scripts/fetch_kdpii_evaluation.py"


def _lossless_pins(compatibility, selected_ids, read, expected, required_pins):
    """Validate an explicit parser amendment without reopening annotations."""
    if (
        not isinstance(compatibility, dict)
        or compatibility.get("version") != "identical_annotation_duplicates_v1"
        or type(compatibility.get("selected_count")) is not int
        or compatibility["selected_count"] != len(selected_ids)
        or type(compatibility.get("dropped_records")) is not int
        or compatibility["dropped_records"] != 0
        or compatibility.get("gold_policy_changed") is not False
    ):
        raise ValueError("Require the approved lossless duplicate parser with the same population")
    pins = {}
    for prefix, path in (
        ("driver", LOSSLESS_FRESH_DRIVER),
        ("helper", LOSSLESS_HELPER),
        ("reference_converter", REFERENCE_CONVERTER),
        ("original_driver", FRESH_DRIVER),
    ):
        name = str(path.resolve())
        sha = digest(path)
        if (
            compatibility.get(prefix + "_path") != name
            or compatibility.get(prefix + "_sha256") != sha
        ):
            raise ValueError("Unrecognized or changed parser source: " + prefix)
        pins[name] = sha
    records = compatibility.get("duplicate_records")
    if not isinstance(records, list) or not records:
        raise ValueError("Require the pre-inference identical-duplicate annotation audit")
    identifiers = []
    for record in records:
        if not isinstance(record, dict) or record.get("id") not in selected_ids:
            raise ValueError("Duplicate audit contains an unselected record")
        identifiers.append(record["id"])
        duplicates = record.get("duplicates")
        if not isinstance(duplicates, list) or not duplicates:
            raise ValueError("Duplicate audit must retain annotation instance indices")
        seen, events = set(), set()
        for row in duplicates:
            if not isinstance(row, dict):
                raise ValueError("Invalid duplicate annotation audit")
            start, end, label, indices = (row.get(k) for k in ("start", "end", "label", "indices"))
            if (
                type(start) is not int or type(end) is not int or not 0 <= start < end
                or not isinstance(label, str) or not label
                or not isinstance(indices, list) or len(indices) < 2
                or any(type(i) is not int or i < 0 for i in indices)
                or indices != sorted(set(indices)) or seen.intersection(indices)
                or (start, end, label) in events
            ):
                raise ValueError("Invalid duplicate annotation coordinates or instance indices")
            seen.update(indices)
            events.add((start, end, label))
    if identifiers != [identifier for identifier in selected_ids if identifier in set(identifiers)]:
        raise ValueError("Duplicate audit repeats or reorders selected records")
    previous_path = Path(compatibility["original_failure_manifest_path"])
    previous_sha = compatibility["original_failure_manifest_sha256"]
    if digest(previous_path) != previous_sha:
        raise ValueError("Original failed pre-inference manifest changed")
    previous = read(previous_path)
    if (
        any(previous.get(key) != value for key, value in expected.items())
        or "sources" in previous or "passed" in previous or "parser_compatibility" in previous
    ):
        raise ValueError("Original failed manifest differs from the fixed pre-inference identity")
    verify_hashes(previous.get("frozen_sha256"))
    if any(previous["frozen_sha256"].get(key) != value for key, value in required_pins.items()):
        raise ValueError("Original failed manifest did not freeze required inputs")
    pins.update(previous["frozen_sha256"])
    pins[str(previous_path.resolve())] = previous_sha
    return pins


def _metric_problem(score):
    problem = metric_problem(score)
    if problem:
        return problem
    if any(type(score.get(k)) is not int for k in ("sentences", "negative_sentences")):
        return "Completed reports require both sentence counts"
    tp, fp, gold = score["tp"], score["fp"], score["gold_names"]
    expected = dict(
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / gold if gold else 0.0,
        full_name_coverage=score["fully_covered_names"] / gold if gold else 0.0,
    )
    for key, value in expected.items():
        actual = score.get(key)
        if (
            type(actual) not in (int, float)
            or not math.isfinite(actual)
            or not math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-12)
        ):
            return f"{key} is missing/nonfinite or disagrees with counts"
    return None


def _runtime_files(policy):
    names = [
        "evaluate_name_multisource_live.py",
        "evaluate_name_generalization.py",
        "name_multisource_adapter.py",
        "name_multisource_selection.py",
        "name_span_evidence.py",
        "name_adapted_adapter.py",
        "name_generalization_model.py",
        "name_bioes_ablation.py",
    ]
    if policy.get("algorithm") == "selective_whole_span":
        names += ["name_selective_adapter.py", "name_selective_evidence.py"]
    return [
        *(ROOT / "scripts" / name for name in names),
        *sorted((ROOT / "src/ko_pii_guard").glob("*.py")),
    ]


def _valid_bootstrap(value):
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("method"), str)
        or type(value.get("repetitions")) is not int
        or value["repetitions"] < 2
    ):
        return False
    intervals = value.get("delta_95_intervals")
    if not isinstance(intervals, dict):
        return False
    for key in ("f1", "precision", "recall", "full_name_coverage"):
        bounds = intervals.get(key)
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in bounds)
            or not -1 <= bounds[0] <= bounds[1] <= 1
        ):
            return False
    return True


def _encoder_pins(report):
    pins = report.get("frozen_sha256")
    if not isinstance(pins, dict):
        return {}
    marker = "/snapshots/" + MODEL_REVISION + "/"
    return {
        path: value
        for path, value in pins.items()
        if isinstance(path, str) and marker in Path(path).as_posix()
    }


def evaluate_release(
    *, decision, development, checkpoint, reserve, live_development, heldout, amendment=None,
    kdpii_lossless_parser=False,
):
    """Return a fail-closed decision from paths to completed, frozen report files."""
    reasons, intervals, input_hashes, report_paths = [], {}, {}, {}
    chosen = None
    amendment_info, amendment_pins, evaluated_ids = None, {}, {}

    def reject(domain, check, detail):
        reasons.append(dict(domain=domain, check=check, detail=str(detail)))

    def outcome():
        return dict(
            scope="multisource external evaluation gate only",
            status="quality_gate_failed" if reasons else "quality_gate_passed",
            quality_gate_passed=not reasons,
            reasons=reasons,
            selected_id=chosen["id"] if chosen else None,
            selected_policy=chosen["policy"] if chosen else None,
            amendment_sha256=(
                input_hashes.get(str(Path(amendment).resolve())) if amendment is not None else None
            ),
            evaluated_selected_counts={domain: len(ids) for domain, ids in evaluated_ids.items()},
            confidence_intervals=intervals,
            input_sha256=input_hashes,
            historical_contracts_checked=False,
            automatic_runtime_promotion=False,
            kdpii_lossless_parser_authorized=kdpii_lossless_parser,
            limitations=[
                "Only the supplied external evaluation gates are checked",
                "Confidence intervals are reported without a new significance gate",
                "Sentence resampling does not establish dialogue independence",
                "Upstream encoder training exposure is not proven absent",
            ],
        )

    def read(path):
        path = Path(path)
        before = digest(path)
        document = read_json(path)
        if not isinstance(document, dict) or digest(path) != before:
            raise ValueError(f"Report must be an unchanged JSON object: {path}")
        input_hashes[str(path.resolve())] = before
        return document

    try:
        decision, development, checkpoint, reserve = map(
            Path, (decision, development, checkpoint, reserve)
        )
        chosen, cached, artifact = validate_selected_decision(decision, development, checkpoint)
        for path in (decision, development):
            input_hashes[str(path.resolve())] = digest(path)
    except FAILURES as exc:
        reject(None, "selected_decision", exc)
        return outcome()  # Never open heldout reports for a rejected development decision.

    try:
        reserves = read(reserve)
        if (
            reserves.get("scope") != "v2_evaluation_reserve_before_any_new_training"
            or reserves.get("test_labels_read") is not False
        ):
            raise ValueError("Require the predeclared ID-only reserve")
        for domain, count in (("klue", 1000), ("kdpii", 2000)):
            ids, consumed = reserves[domain]["selected_ids"], reserves[domain]["consumed_ids"]
            if (
                not isinstance(ids, list)
                or any(not isinstance(x, str) or not x for x in ids)
                or len(ids) != count
                or len(set(ids)) != count
                or not isinstance(consumed, list)
                or set(ids) & set(consumed)
            ):
                raise ValueError(
                    f"{domain}: invalid reserve count, duplicates, or consumed overlap"
                )
            evaluated_ids[domain] = ids
        verify_hashes(reserves.get("source_sha256"))
        exclusion = Path(reserves["exclusion_hashes_path"])
        if (
            digest(exclusion) != reserves.get("exclusion_hashes_sha256")
            or read_json(checkpoint / "data-manifest.json")
            .get("input_sha256", {})
            .get(str(exclusion))
            != reserves["exclusion_hashes_sha256"]
        ):
            raise ValueError("Reserve exclusions differ from frozen training preparation")
    except FAILURES as exc:
        reject(None, "reserve_provenance", exc)
        return outcome()

    def reports(paths, stage):
        if not isinstance(paths, (list, tuple)) or len(paths) != 2:
            raise ValueError(f"Require exactly both {stage} domain reports")
        rows = [read(path) for path in paths]
        report_paths.update({id(row): Path(path) for row, path in zip(rows, paths, strict=True)})
        if {row.get("domain") for row in rows} != {"klue", "kdpii"}:
            raise ValueError(f"Require distinct KLUE and KDPII {stage} reports")
        return rows

    try:
        live = reports(live_development, "live development")
        signature = live[0].get("runtime_signature")
        if (
            not isinstance(signature, dict)
            or any(
                not isinstance(signature.get(key), str) or not signature[key]
                for key in ("python", "torch")
            )
            or not isinstance(signature.get("packages"), dict)
            or any(
                not isinstance(signature["packages"].get(key), str)
                or not signature["packages"][key]
                for key in ("transformers", "presidio-analyzer", "safetensors")
            )
        ):
            raise ValueError("Require recorded evaluator runtime versions")
        identity = dict(
            decision_sha256=digest(decision),
            development_report_sha256=digest(development),
            artifact_sha256=artifact,
            selected_policy=chosen["policy"],
            reserve_sha256=digest(reserve),
            device="cpu",
            runtime_signature=signature,
        )
        validate_live_reports(live, identity)
    except FAILURES as exc:
        reject(None, "live_development", exc)
        return outcome()

    required_pins = {**cached["frozen_sha256"], **artifact}
    try:
        required_pins.update(
            {
                str(path.resolve()): digest(path)
                for path in [
                    decision,
                    development,
                    reserve,
                    exclusion,
                    *_runtime_files(chosen["policy"]),
                ]
            }
        )
    except FAILURES as exc:
        reject(None, "frozen_inputs", exc)
        return outcome()
    factory = (
        "name_selective_adapter:build_ner"
        if chosen["policy"].get("algorithm") == "selective_whole_span"
        else "name_multisource_adapter:build_ner"
    )
    encoder_pins = _encoder_pins(live[0])
    if not any(Path(path).suffix == ".safetensors" for path in encoder_pins) or not any(
        Path(path).name == "config.json" for path in encoder_pins
    ):
        reject(None, "encoder_snapshot", "Pinned original encoder weights/configuration are absent")

    def check_report(report, stage):
        domain = report["domain"]
        context = stage + "/" + domain
        if (
            report.get("scope") != "multisource_live_" + stage
            or report.get("passed") is not True
            or report.get("source_changed_during_run") is not False
            or report.get("changed_inputs") != []
            or report.get("resampling") is not False
            or report.get("runtime_promotion") is not False
            or report.get("evaluation_blocked")
        ):
            reject(
                context, "completed_report", "Incomplete/failed run, changed inputs, or resampling"
            )
        expected = dict(
            identity,
            selected_id=chosen["id"],
            candidate_factory=factory,
            model_kind="frozen",
            guard_score_threshold=0.0,
            baseline_model=dict(model_id=MODEL_ID, revision=MODEL_REVISION, threshold=0.9),
        )
        for key, value in expected.items():
            if report.get(key) != value:
                reject(context, "same_candidate", f"Report differs from selected {key}")
        if report.get("runtime") != dict(signature, devices=dict(baseline="cpu", candidate="cpu")):
            reject(context, "runtime", "Actual devices or runtime versions differ")
        if _encoder_pins(report) != encoder_pins:
            reject(context, "encoder_snapshot", "Pinned original encoder snapshot differs")
        if report.get("ancestry_overlap") != dict(exact_ids=[], normalized_ids=[]):
            reject(
                context, "training_overlap", "Exact/normalized ancestry checks missing or nonempty"
            )
        amendment_fields = {}
        if stage == "heldout":
            if amendment_info is not None:
                audited = amendment_info["domains"][domain]
                amendment_fields = dict(
                    amendment_sha256=digest(amendment),
                    amendment_path=str(Path(amendment).resolve()),
                    amendment_created_at_utc=amendment_info["created_at_utc"],
                    original_selected_ids=audited["original_selected_ids"],
                    excluded_ids=audited["excluded_ids"],
                    original_selected_count=audited["original_count"],
                    selected_count=audited["selected_count"],
                    excluded_count=audited["excluded_count"],
                )
                if any(report.get(key) != value for key, value in amendment_fields.items()):
                    reject(
                        context, "amendment_provenance", "Report differs from validated amendment"
                    )
            elif report.get("amendment_sha256") is not None:
                reject(context, "amendment_provenance", "Amended reports require --amendment")
        parser_pins, parser_fields = {}, {}
        parser_required = kdpii_lossless_parser and stage == "heldout" and domain == "kdpii"
        report_pins = report.get("frozen_sha256")
        if (
            parser_required or "parser_compatibility" in report
            or (isinstance(report_pins, dict)
                and str(LOSSLESS_FRESH_DRIVER.resolve()) in report_pins)
        ):
            try:
                if not parser_required or amendment_info is None:
                    raise ValueError(
                        "Lossless KDPII parser requires explicit authorization and amendment"
                    )
                parser_fields = dict(parser_compatibility=report.get("parser_compatibility"))
                parser_pins = _lossless_pins(
                    report.get("parser_compatibility"), evaluated_ids[domain], read,
                    dict(expected, **amendment_fields, selected_ids=evaluated_ids[domain],
                         scope="multisource_live_heldout", domain=domain),
                    {**required_pins, **encoder_pins, **amendment_pins,
                     **reserves["source_sha256"],
                     **{str(Path(path).resolve()): digest(path) for path in live_development}},
                )
            except FAILURES as exc:
                reject(context, "parser_compatibility", exc)
        try:
            pins = report.get("frozen_sha256")
            verify_hashes(pins)
            wanted = dict(required_pins)
            if stage == "heldout":
                wanted.update(reserves["source_sha256"])
                wanted.update(amendment_pins)
                wanted.update(parser_pins)
                wanted.update(
                    {str(Path(path).resolve()): digest(path) for path in live_development}
                )
            if any(pins.get(path) != value for path, value in wanted.items()):
                raise ValueError(
                    "Required selected/runtime/source/predecessor inputs are not pinned"
                )
        except FAILURES as exc:
            reject(context, "frozen_inputs", exc)
        ids = report.get("selected_ids")
        valid_ids = (
            isinstance(ids, list)
            and all(isinstance(x, str) for x in ids)
            and len(ids) == len(set(ids))
        )
        if valid_ids:
            if stage == "heldout":
                valid_ids = ids == evaluated_ids[domain]
            else:
                valid_ids = len(ids) == (1000 if domain == "klue" else 500) and set(ids) <= set(
                    reserves[domain]["consumed_ids"]
                )
        if not valid_ids:
            reject(
                context, "fixed_population", "Selected IDs differ, repeat, or require resampling"
            )
            ids = []
        if stage == "heldout" and amendment_info is not None:
            try:
                manifest_path = Path(report["pre_inference_manifest_path"])
                manifest_sha = report["pre_inference_manifest_sha256"]
                if (
                    manifest_path.resolve()
                    != report_paths[id(report)].with_suffix(".manifest.json").resolve()
                    or digest(manifest_path) != manifest_sha
                    or report.get("frozen_sha256", {}).get(str(manifest_path.resolve()))
                    != manifest_sha
                ):
                    raise ValueError("Pre-inference manifest is absent, changed, or unpinned")
                manifest = read(manifest_path)
                wanted = dict(
                    expected,
                    **amendment_fields,
                    **parser_fields,
                    selected_ids=evaluated_ids[domain],
                    scope="multisource_live_heldout",
                    domain=domain,
                )
                if any(manifest.get(key) != value for key, value in wanted.items()):
                    raise ValueError(
                        "Pre-inference manifest differs from completed report identity"
                    )
                pins = manifest.get("frozen_sha256")
                verify_hashes(pins)
                before_inference = {
                    **required_pins,
                    **encoder_pins,
                    **amendment_pins,
                    **parser_pins,
                    **reserves["source_sha256"],
                    **{str(Path(path).resolve()): digest(path) for path in live_development},
                }
                if any(pins.get(key) != value for key, value in before_inference.items()):
                    raise ValueError("Required inputs were not frozen before inference")
                if any(report["frozen_sha256"].get(key) != value for key, value in pins.items()):
                    raise ValueError("Manifest and completed report frozen inputs differ")
            except FAILURES as exc:
                reject(context, "pre_inference_amendment", exc)
        source_rows = report.get("sources")
        if not isinstance(source_rows, dict) or set(source_rows) != set(DOMAIN_SOURCES[domain]):
            reject(
                context,
                "source_complete",
                "Every domain source/policy must be present exactly once",
            )
            return
        for name, row in source_rows.items():
            if not isinstance(row, dict):
                reject(name, "source_complete", "Missing source result")
                continue
            if not strict_gate_passes(row.get("regression_gate")):
                reject(
                    name, "strict_regression_gate", "Missing/failed concrete per-case masking gate"
                )
            if row.get("passed") is not True or row.get("aggregate_failures") != []:
                reject(name, "aggregate_constraints", "Source reports a failed quality gate")
            old, new = row.get("baseline"), row.get("metrics")
            count = (
                sum(identifier.endswith("-" + name) for identifier in ids)
                if domain == "klue"
                else len(ids)
            )
            valid = True
            for label, score in (("baseline", old), ("candidate", new)):
                problem = _metric_problem(score)
                if problem:
                    reject(name, "valid_metrics", label + ": " + problem)
                    valid = False
                elif score["sentences"] != count or count == 0:
                    reject(
                        name,
                        "fixed_population",
                        label + ": sentence count differs from selected IDs",
                    )
            if valid:
                if any(
                    new[key] != old[key]
                    for key in ("gold_names", "negative_sentences", "sentences")
                ):
                    reject(
                        name,
                        "same_population",
                        "Candidate and baseline evaluated different populations",
                    )
                worse = [key for key in HIGHER if new[key] < old[key]]
                worse += [key for key in LOWER if new[key] > old[key]]
                if worse:
                    reject(
                        name, "aggregate_constraints", "Worse than baseline: " + ", ".join(worse)
                    )
                if stage == "development":
                    expected_scores = (
                        cached["baselines"][name],
                        chosen["sources"][name]["metrics"],
                    )
                    if any(
                        compact(actual) != compact(expected_score)
                        for actual, expected_score in zip((old, new), expected_scores, strict=True)
                    ):
                        reject(
                            name, "live_replay", "Live metrics differ from selected cached metrics"
                        )
            if not _valid_bootstrap(row.get("bootstrap")):
                reject(
                    name, "confidence_intervals", "Completed bootstrap intervals missing or invalid"
                )
            elif stage == "heldout":
                intervals[name] = row["bootstrap"]
        if domain == "klue" and sum(x.endswith(("-nsmc", "-wikitree")) for x in ids) != len(ids):
            reject(context, "fixed_population", "Selected KLUE IDs have unrecognized sources")
        if stage == "development":
            replay = report.get("cache_replay", {})
            if replay.get("matched") is not True or replay.get("metric_differences") != []:
                reject(context, "live_replay", "Development cache metrics do not match")
            for label in ("baseline", "candidate"):
                comparison = replay.get(label, {})
                if (
                    comparison.get("matched") is not True
                    or comparison.get("span_disagreement_ids") != []
                    or comparison.get("mask_disagreement_ids") != []
                ):
                    reject(
                        context, "live_replay", label + ": coordinates or actual stars do not match"
                    )

    for report in live:
        check_report(report, "development")
    if reasons:
        return outcome()
    if amendment is not None:
        try:
            amendment = Path(amendment)
            recorded = read(amendment)
            auditor = importlib.import_module("name_reserve_text_audit")
            amendment_info = auditor.validate_amendment(amendment, reserve)
            if (
                amendment_info != recorded
                or amendment_info.get("scope") != "multisource_reserve_text_amendment"
                or amendment_info.get("original_reserve_sha256") != digest(reserve)
                or not isinstance(amendment_info.get("created_at_utc"), str)
            ):
                raise ValueError(
                    "Amendment identity differs from original reserve or recorded audit"
                )
            for domain in DOMAIN_SOURCES:
                audited = amendment_info["domains"][domain]
                original = reserves[domain]["selected_ids"]
                kept, excluded = audited["selected_ids"], audited["excluded_ids"]
                if (
                    audited.get("original_selected_ids") != original
                    or not isinstance(kept, list)
                    or not isinstance(excluded, list)
                    or any(not isinstance(x, str) for x in [*kept, *excluded])
                    or len(set(excluded)) != len(excluded)
                    or not set(excluded) <= set(original)
                    or kept != [x for x in original if x not in set(excluded)]
                    or audited.get("original_count") != len(original)
                    or audited.get("selected_count") != len(kept)
                    or audited.get("excluded_count") != len(excluded)
                ):
                    raise ValueError("Amendment must only exclude audited original IDs in order")
                evaluated_ids[domain] = kept
            amendment_pins = {
                str(path.resolve()): digest(path)
                for path in [amendment, Path(auditor.__file__), FRESH_DRIVER]
            }
        except FAILURES as exc:
            reject(None, "amendment_provenance", exc)
            return outcome()
    try:
        fresh = reports(heldout, "heldout")
    except FAILURES as exc:
        reject(None, "completed_report", exc)
        return outcome()
    for report in fresh:
        check_report(report, "heldout")
    # Reuse exactly the development acceptance conditions on ONE already fixed
    # candidate. No search, retuning, or pooled improvement can waive a source.
    baselines, sources = {}, {}
    for report in fresh:
        rows = report.get("sources")
        if not isinstance(rows, dict):
            continue
        for name, row in rows.items():
            if isinstance(row, dict):
                baselines[name] = row.get("baseline")
                sources[name] = dict(
                    metrics=row.get("metrics"), regression_gate=row.get("regression_gate")
                )
    quality = select_candidates(
        baselines,
        [dict(id=chosen["id"], policy=chosen["policy"], sources=sources)],
        required_sources=REQUIRED_SOURCES,
    )
    for detail in quality["global_reasons"]:
        reject(None, "fresh_quality", detail)
    for reason in quality["rejections"].get(chosen["id"], []):
        reject(reason["source"], reason["check"], reason["detail"])
    try:
        verify_hashes(input_hashes)
    except FAILURES as exc:
        reject(None, "inputs_changed_during_check", exc)
    return outcome()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("decision", "development", "checkpoint", "reserve"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--live-development", type=Path, nargs=2, required=True, metavar=("KLUE", "KDPII")
    )
    parser.add_argument("--heldout", type=Path, nargs=2, required=True, metavar=("KLUE", "KDPII"))
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--kdpii-lossless-parser", action="store_true",
                        help="Verify the separately frozen identical-annotation-duplicate parser")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Require a new output path; previous decisions must not be overwritten")
    result = evaluate_release(
        decision=args.decision,
        development=args.development,
        checkpoint=args.checkpoint,
        reserve=args.reserve,
        live_development=args.live_development,
        heldout=args.heldout,
        amendment=args.amendment,
        kdpii_lossless_parser=args.kdpii_lossless_parser,
    )
    result["checker_sha256"] = digest(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.check and not result["quality_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
