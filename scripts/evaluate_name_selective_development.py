"""Evaluate a fixed selective span grid on previously consumed development IDs.

Baseline anchors are removed only when the complete anchor interval is O with
sufficient posterior evidence. Additions require complete-name span evidence.
No unseen reserve is read and no runtime model is promoted by this command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diagnose_name_generalization import reconstruct
from evaluate_name_generalization import actual_mask_gate, evaluate_guard, read_selected_cases
from evaluate_name_span_evidence_development import (
    assert_baseline_replay,
    compact,
    digest,
    guard_for,
    validate_cache_cases,
    validate_cache_provenance,
    validate_pinned_file,
)
from name_span_evidence import span_posteriors
from presidio_analyzer import RecognizerResult

ADDITIONS = (.99, .999, .9999, 1.0)
REMOVALS = (.9, .99, .999, .9999, 1.0)
REQUIRED_SOURCES = ("nsmc", "wikitree", "kdpii/PS_NAME", "kdpii/PS_NAME+PS_NICKNAME")


def policy_grid():
    return [dict(algorithm="selective_whole_span", addition_threshold=addition,
                 removal_threshold=removal, class_logit_correction=0.0)
            for addition in ADDITIONS for removal in REMOVALS]


def rows_from_selected(selected_rows, retained_anchors, addresses):
    """Address exclusions apply to proposals, while retained baseline anchors survive."""
    if len(selected_rows) != len(retained_anchors) or len(addresses) != len(selected_rows):
        raise ValueError("Selected spans, anchors and addresses must have matching row counts")
    rows = []
    for spans, retained, address_row in zip(
            selected_rows, retained_anchors, addresses, strict=True):
        names = [RecognizerResult("KR_NAME", start, end, score)
                 for start, end, score in spans
                 if (start, end) in retained or not any(
                     start < address.end and address.start < end for address in address_row)]
        rows.append([*address_row, *names])
    return rows


def assert_identity_policy(score, predictions, masks, baseline_score,
                           baseline_predictions, baseline_masks, domain):
    assert_baseline_replay(score, predictions, baseline_predictions, baseline_score, domain)
    if masks != baseline_masks:
        raise ValueError(f"Identity selective policy changed actual stars masks: {domain}")


def domains_for(klue, kdpii):
    domains = {}
    for source in REQUIRED_SOURCES[:2]:
        ids = [i for i, case in enumerate(klue) if case["id"].endswith("-" + source)]
        if not ids:
            raise ValueError(f"Missing required KLUE source: {source}")
        domains[source] = ids, [klue[i] for i in ids]
    if sum(len(ids) for ids, _ in domains.values()) != len(klue) or not kdpii:
        raise ValueError("Every consumed case must belong to a required source")
    kd_ids = list(range(len(klue), len(klue) + len(kdpii)))
    domains["kdpii/PS_NAME"] = kd_ids, kdpii
    domains["kdpii/PS_NAME+PS_NICKNAME"] = kd_ids, [
        dict(case, expected=case["alternate_expected"]) for case in kdpii]
    return domains


def validate_cache_collection(cache, provenance, cache_path, spans_path, domain):
    if (cache.get("scope") != "consumed_development_cache"
            or cache.get("source_changed_during_run") is not False):
        raise ValueError("Require completed consumed development cache")
    if cache.get("domain") != domain:
        raise ValueError("Cache domain differs from its requested source")
    validate_cache_provenance(provenance, cache_path, spans_path)
    raw = [json.loads(line) for line in spans_path.read_text().splitlines()]
    if [row["id"] for row in raw] != cache.get("selected_ids"):
        raise ValueError("Raw spans differ from consumed cache IDs")
    spans = [row["spans"] for row in raw]
    if spans != cache.get("spans"):
        raise ValueError("Separate raw spans differ from embedded frozen cache spans")
    return spans


def main():
    from name_selective_evidence import all_o_posteriors, select_selective_evidence

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--klue-cache", type=Path,
                        default=Path("/tmp/name-multisource-v2-trained-klue.pt"))
    parser.add_argument("--klue-spans", type=Path,
                        default=Path("/tmp/name-multisource-v2-trained-klue.spans.jsonl"))
    parser.add_argument("--klue-cache-provenance", type=Path,
                        default=Path("/tmp/name-multisource-v2-trained-klue.json"))
    parser.add_argument("--klue-report", type=Path, default=Path(
        "benchmarks/results/name-generalization-final-v1-nonoverlap-development.json"))
    parser.add_argument("--klue-source", type=Path,
                        default=Path("/tmp/ko-pii-klue-ner/klue-ner-v1.1_dev.tsv"))
    parser.add_argument("--kdpii-cache", type=Path,
                        default=Path("/tmp/name-multisource-v2-trained-kdpii.pt"))
    parser.add_argument("--kdpii-spans", type=Path,
                        default=Path("/tmp/name-multisource-v2-trained-kdpii.spans.jsonl"))
    parser.add_argument("--kdpii-cache-provenance", type=Path,
                        default=Path("/tmp/name-multisource-v2-trained-kdpii.json"))
    parser.add_argument("--kdpii-report", type=Path, default=Path(
        "benchmarks/results/name-generalization-final-v1-nonoverlap-kdpii.json"))
    parser.add_argument("--kdpii-cases", type=Path,
                        default=Path("/tmp/ko-pii-kdpii-v1/cases.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        parser.error("Preserve prior results; require unused output paths")
    if args.output.resolve() == manifest_path.resolve():
        parser.error("Report and manifest output paths must be distinct")
    torch.set_num_threads(2)
    reports = [json.loads(path.read_text()) for path in (args.klue_report, args.kdpii_report)]
    if (reports[0].get("scope") != "development"
            or reports[1].get("scope") != "second_source_KDPII_frozen_candidate"
            or any(report.get("source_changed_during_run") is not False for report in reports)):
        raise ValueError("Only completed, explicitly consumed development reports are allowed")
    caches = [torch.load(path, map_location="cpu", weights_only=True)
              for path in (args.klue_cache, args.kdpii_cache)]
    provenance = [json.loads(path.read_text()) for path in (
        args.klue_cache_provenance, args.kdpii_cache_provenance)]
    raw_spans = []
    for report, cache, proof, path, spans, domain in zip(
            reports, caches, provenance, (args.klue_cache, args.kdpii_cache),
            (args.klue_spans, args.kdpii_spans), ("klue", "kdpii"), strict=True):
        if report["selected_ids"] != cache["selected_ids"]:
            raise ValueError("Cache IDs differ from already consumed report")
        raw_spans.extend(validate_cache_collection(cache, proof, path, spans, domain))
        for document in (report, cache):
            if not document.get("frozen_sha256"):
                raise ValueError("Completed report/cache must pin its complete source provenance")
            for frozen_path, expected in document["frozen_sha256"].items():
                if digest(frozen_path) != expected:
                    raise ValueError(f"Consumed input changed: {frozen_path}")
    for report, cache, path in zip(reports, caches, (args.klue_source, args.kdpii_cases),
                                    strict=True):
        validate_pinned_file(report, path)
        validate_pinned_file(cache, path)
    klue = read_selected_cases(args.klue_source, reports[0]["selected_ids"])
    kdpii = [json.loads(line) for line in args.kdpii_cases.read_text().splitlines()]
    for cache, cases in zip(caches, (klue, kdpii), strict=True):
        validate_cache_cases(cache, cases)
    originals = [reconstruct(klue, reports[0]["baseline"]["metrics"]),
                 reconstruct(kdpii, reports[1]["baseline"]["metrics"]["PS_NAME"])]
    original = originals[0] + originals[1]
    logits = caches[0]["logits"] + caches[1]["logits"]
    addresses = [[RecognizerResult(row["entity"], row["start"], row["end"], row["score"])
                  for row in spans if row["entity"] == "KR_ADDRESS"] for spans in raw_spans]
    domains = domains_for(klue, kdpii)
    files = [Path(__file__), *[value for value in vars(args).values()
                              if isinstance(value, Path) and value != args.output],
             *[Path(__file__).with_name(name) for name in (
                 "name_selective_evidence.py", "name_span_evidence.py",
                 "evaluate_name_span_evidence_development.py", "name_multisource_selection.py",
                 "name_bioes_ablation.py", "evaluate_name_generalization.py",
                 "diagnose_name_generalization.py", "train_name_span_experiment.py")],
             *Path("src/ko_pii_guard").glob("*.py")]
    hashes = {path: expected for document in [*reports, *caches]
              for path, expected in document["frozen_sha256"].items()}
    hashes.update({str(path.resolve()): digest(path) for path in files})
    policies = policy_grid()
    manifest = dict(scope="multisource_development_only", algorithm="selective_whole_span",
                    policies=policies, addition_thresholds=ADDITIONS, removal_thresholds=REMOVALS,
                    class_logit_correction=0.0, required_sources=list(REQUIRED_SOURCES),
                    protect_anchors="Except anchors with P(all O over full interval) > removal",
                    address_policy="Retained baseline anchors survive; proposals use address veto",
                    inputs="Frozen multisource-v2-trained CPU FP32 consumed KLUE1000/KDPII500",
                    frozen_sha256=hashes, independent_evaluation=False, runtime_promotion=False,
                    duplicate_text_policy="Every ID retained; identical outputs for repeated text",
                    limitations="Cached-logit selective ablation; no live parity or independent "
                    "accuracy claim. Posterior confidence is model-relative, not calibrated risk.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    # Freeze grid and transitive source hashes before replay or posterior computation.
    baseline_rows = [[RecognizerResult("KR_NAME", start, end, 1.) for start, end in spans]
                     for spans in original]
    score, predictions, _ = evaluate_guard(klue, guard_for(klue, baseline_rows[:len(klue)]))
    assert_baseline_replay(score, predictions, originals[0],
                           reports[0]["baseline"]["metrics"], "klue/all")
    expected = {**reports[0]["baseline"]["metrics"]["by_source"],
                **{f"kdpii/{policy}": metrics for policy, metrics in
                   reports[1]["baseline"]["metrics"].items()}}
    baselines, baseline_predictions, baseline_masks = {}, {}, {}
    for domain, (indices, cases) in domains.items():
        score, predictions, masks = evaluate_guard(
            cases, guard_for(cases, [baseline_rows[i] for i in indices]))
        assert_baseline_replay(score, predictions, [original[i] for i in indices],
                               expected[domain], domain)
        baselines[domain] = compact(score)
        baseline_predictions[domain], baseline_masks[domain] = predictions, masks
    anchors = [[(start, end, 1.) for start, end in sorted(spans)] for spans in original]
    posterior = [span_posteriors(value) for value in logits]
    outside = [all_o_posteriors(value, [(start, end) for start, end, _ in row])
               for value, row in zip(logits, anchors, strict=True)]
    results, identity_verified = [], False
    for policy in policies:
        addition, removal = policy["addition_threshold"], policy["removal_threshold"]
        selected = [select_selective_evidence(span, out, addition, removal, row)
                    for span, out, row in zip(posterior, outside, anchors, strict=True)]
        retained = [{(start, end) for (start, end, _), probability in zip(row, out, strict=True)
                     if float(probability) <= removal}
                    for row, out in zip(anchors, outside, strict=True)]
        predictions_by_case = rows_from_selected(selected, retained, addresses)
        evaluations = {}
        for domain, (indices, cases) in domains.items():
            score, predictions, masks = evaluate_guard(
                cases, guard_for(cases, [predictions_by_case[i] for i in indices]))
            if addition == removal == 1.0:
                assert_identity_policy(score, predictions, masks, baselines[domain],
                                       baseline_predictions[domain], baseline_masks[domain], domain)
                identity_verified = True
            evaluations[domain] = dict(metrics=compact(score), regression_gate=actual_mask_gate(
                cases, baseline_predictions[domain], predictions, baseline_masks[domain], masks))
        identifier = f"selective:addition={addition}:removal={removal}"
        results.append(dict(id=identifier, policy=policy, sources=evaluations))
        print(json.dumps(dict(policy=policy, sources={source: dict(
            tp=result["metrics"]["tp"], fp=result["metrics"]["fp"], fn=result["metrics"]["fn"],
            full=result["metrics"]["fully_covered_names"], gate=result["regression_gate"]["passed"])
            for source, result in evaluations.items()})), flush=True)
    changed = any(digest(path) != expected for path, expected in hashes.items())
    report = dict(**manifest, baselines=baselines, candidates=results,
                  baseline_replay_verified=True, identity_policy_verified=identity_verified,
                  source_changed_during_run=changed)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if changed:
        raise ValueError("Frozen selective development inputs changed during execution")


if __name__ == "__main__":
    main()
