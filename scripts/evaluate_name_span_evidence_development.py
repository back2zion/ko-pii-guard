"""Compare fixed whole-span policies on explicitly consumed development only.

The supplied model caches are immutable. This is a decoding ablation, not a
re-evaluation on an independent test set. Each source/policy must pass separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from diagnose_name_generalization import reconstruct
from evaluate_name_generalization import actual_mask_gate, evaluate_guard, read_selected_cases
from name_span_evidence import select_span_evidence, span_posteriors
from presidio_analyzer import RecognizerResult

from ko_pii_guard import KoreanPIIGuard
from ko_pii_guard.normalization import normalize_text

CUTOFFS = (.01, .025, .05, .1, .2, .3, .5, .7, .9, .99)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_pinned_file(document, path, key="frozen_sha256"):
    expected = document.get(key, {}).get(str(path.resolve()))
    if expected != digest(path):
        raise ValueError(f"Input is absent from or differs from consumed provenance: {path}")


def validate_cache_provenance(provenance, cache_path, spans_path):
    if (provenance.get("scope") not in (
            "development_posterior_coverage_selection", "consumed_development_cache")
            or provenance.get("source_changed_during_run") is not False):
        raise ValueError("Require completed development provenance for KLUE cache and spans")
    for path in (cache_path, spans_path):
        validate_pinned_file(provenance, path, key="source_sha256")


def validate_cache_cases(cache, cases):
    identifiers = [case["id"] for case in cases]
    if cache["selected_ids"] != identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Cache IDs differ from unique consumed case IDs")
    if any(normalize_text(case["text"]).text != case["text"] or not case["text"].strip()
           for case in cases):
        raise ValueError("Replay requires nonempty identity normalization")
    logits = cache["logits"]
    if len(logits) != len(cases) or any(
        not isinstance(value, torch.Tensor) or value.shape != (len(case["text"]), 5)
        or not value.is_floating_point() or not torch.isfinite(value).all()
        for case, value in zip(cases, logits, strict=True)
    ):
        raise ValueError("Cache logits must completely cover each original text")


class ReplayNER:
    def __init__(self, cases, rows):
        if len(cases) != len(rows):
            raise ValueError("One prediction row is required for every consumed ID")
        self.lookup, signatures = {}, {}
        for case, row in zip(cases, rows, strict=True):
            signature = sorted((r.entity_type, r.start, r.end, r.score) for r in row)
            text = case["text"]
            if text in signatures and signatures[text] != signature:
                raise ValueError("Conflicting predictions for duplicate development text")
            self.lookup[text], signatures[text] = row, signature

    def analyze(self, text):
        return self.lookup[text]


def guard_for(cases, rows):
    return KoreanPIIGuard(entities=["KR_NAME"], score_threshold=0., ner=ReplayNER(cases, rows))


def compact(metrics):
    return {key: value for key, value in metrics.items() if key not in ("errors", "by_source")}


def assert_baseline_replay(score, predictions, expected_predictions, expected_metrics, domain):
    if predictions != expected_predictions:
        raise ValueError(f"Baseline replay changed per-case predictions: {domain}")
    for key, expected in compact(expected_metrics).items():
        if score.get(key) != expected:
            raise ValueError(f"Baseline replay differs from consumed {domain} metric {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--klue-cache", type=Path,
                        default=Path("/tmp/name-generalization-lora-v1-development-logits.pt"))
    parser.add_argument("--klue-spans", type=Path, default=Path(
        "benchmarks/results/name-generalization-lora-v1-development.spans.jsonl"))
    parser.add_argument("--klue-report", type=Path, default=Path(
        "benchmarks/results/name-generalization-final-v1-nonoverlap-development.json"))
    parser.add_argument("--klue-cache-provenance", type=Path, default=Path(
        "benchmarks/results/name-generalization-lora-v1-coverage-selection.json"))
    parser.add_argument("--klue-source", type=Path,
                        default=Path("/tmp/ko-pii-klue-ner/klue-ner-v1.1_dev.tsv"))
    parser.add_argument("--kdpii-cache", type=Path,
                        default=Path("/tmp/name-multisource-v2-kdpii-development-logits.pt"))
    parser.add_argument("--kdpii-cases", type=Path,
                        default=Path("/tmp/ko-pii-kdpii-v1/cases.jsonl"))
    parser.add_argument("--kdpii-report", type=Path, default=Path(
        "benchmarks/results/name-generalization-final-v1-nonoverlap-kdpii.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--class-logit-corrections", type=float, nargs="+",
                        default=[0., math.log(4.)])
    parser.add_argument("--inputs-description", default=(
        "KLUE1000 CUDA BF16 adapted logits; consumed KDPII500 CPU FP32 logits"))
    args = parser.parse_args()
    if (any(not math.isfinite(value) for value in args.class_logit_corrections)
            or len(set(args.class_logit_corrections)) != len(args.class_logit_corrections)):
        parser.error("Class logit corrections must be finite distinct values")
    if not args.inputs_description.strip():
        parser.error("Require an explicit description of cache runtime provenance")
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        parser.error("Preserve all prior results; require unused output paths")
    torch.set_num_threads(2)
    reports = [json.loads(p.read_text()) for p in (args.klue_report, args.kdpii_report)]
    if (reports[0]["scope"] != "development"
            or reports[1]["scope"] != "second_source_KDPII_frozen_candidate"
            or any(r["source_changed_during_run"] for r in reports)):
        raise ValueError("Only explicitly consumed, completed sources may be compared")
    caches = [torch.load(p, weights_only=True) for p in (args.klue_cache, args.kdpii_cache)]
    for report, cache in zip(reports, caches, strict=True):
        if report["selected_ids"] != cache["selected_ids"]:
            raise ValueError("Cache IDs differ from already consumed evaluation")
        for path, expected in report["frozen_sha256"].items():
            if digest(path) != expected:
                raise ValueError(f"Consumed evaluation input changed: {path}")
        for path, expected in cache["frozen_sha256"].items():
            if digest(path) != expected:
                raise ValueError(f"Cached inference source changed: {path}")
        if cache.get("source_changed_during_run", False) is not False:
            raise ValueError("Cache inputs changed during collection")
    provenance = json.loads(args.klue_cache_provenance.read_text())
    validate_cache_provenance(provenance, args.klue_cache, args.klue_spans)
    for report, cache, path in zip(reports, caches, (args.klue_source, args.kdpii_cases),
                                    strict=True):
        validate_pinned_file(report, path)
        validate_pinned_file(cache, path)
    cases = read_selected_cases(args.klue_source, reports[0]["selected_ids"])
    kd = [json.loads(line) for line in args.kdpii_cases.read_text().splitlines()]
    if [c["id"] for c in kd] != reports[1]["selected_ids"]:
        raise ValueError("KDPII records differ from consumed IDs")
    for cache, rows in zip(caches, (cases, kd), strict=True):
        validate_cache_cases(cache, rows)
    if len(caches[1]["spans"]) != len(kd):
        raise ValueError("KDPII raw spans do not cover all consumed cases")
    raw = [json.loads(line) for line in args.klue_spans.read_text().splitlines()]
    if [r["id"] for r in raw] != reports[0]["selected_ids"]:
        raise ValueError("KLUE raw outputs differ from consumed IDs")
    klue_original = reconstruct(cases, reports[0]["baseline"]["metrics"])
    kd_original = reconstruct(kd, reports[1]["baseline"]["metrics"]["PS_NAME"])
    all_cases = cases + kd
    original = klue_original + kd_original
    logits = caches[0]["logits"] + caches[1]["logits"]
    raw_spans = [r["spans"] for r in raw] + caches[1]["spans"]
    addresses = [[RecognizerResult(r["entity"], r["start"], r["end"], r["score"])
                  for r in row if r["entity"] == "KR_ADDRESS"] for row in raw_spans]
    domains = {}
    for source in ("nsmc", "wikitree"):
        ids = [i for i, c in enumerate(cases) if c["id"].endswith("-" + source)]
        domains[source] = (ids, [all_cases[i] for i in ids])
    kd_ids = list(range(len(cases), len(all_cases)))
    domains["kdpii/PS_NAME"] = kd_ids, kd
    domains["kdpii/PS_NAME+PS_NICKNAME"] = kd_ids, [
        dict(c, expected=c["alternate_expected"]) for c in kd]
    files = [Path(__file__), *[v for v in vars(args).values()
                              if isinstance(v, Path) and v != args.output],
             *[Path(__file__).with_name(name) for name in (
                 "name_span_evidence.py", "name_multisource_selection.py",
                 "name_bioes_ablation.py",
                 "evaluate_name_generalization.py", "diagnose_name_generalization.py",
                 "train_name_span_experiment.py")], *Path("src/ko_pii_guard").glob("*.py")]
    hashes = {path: expected for document in [*reports, *caches]
              for path, expected in document["frozen_sha256"].items()}
    hashes.update({str(path.resolve()): digest(path) for path in files})
    manifest = dict(scope="multisource_development_only", cutoffs=CUTOFFS,
                    class_logit_corrections=args.class_logit_corrections,
                    protect_anchors=[False, True],
                    required_sources=list(domains), frozen_sha256=hashes,
                    inputs=args.inputs_description,
                    limitations="Cached-logit ablation; no live candidate parity or "
                                "independent accuracy claim",
                    duplicate_text_policy="Retain every ID; require identical outputs "
                                          "for repeated text",
                    independent_evaluation=False, runtime_promotion=False)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    baselines, baseline_predictions, baseline_masks = {}, {}, {}
    baseline_rows = [[RecognizerResult("KR_NAME", s, e, 1.) for s, e in spans]
                     for spans in original]
    score, predictions, _ = evaluate_guard(cases, guard_for(cases, baseline_rows[:len(cases)]))
    assert_baseline_replay(score, predictions, klue_original,
                           reports[0]["baseline"]["metrics"], "klue/all")
    expected_domains = {
        **reports[0]["baseline"]["metrics"]["by_source"],
        **{f"kdpii/{policy}": score for policy, score
           in reports[1]["baseline"]["metrics"].items()},
    }
    for domain, (ids, rows) in domains.items():
        score, predictions, masks = evaluate_guard(
            rows, guard_for(rows, [baseline_rows[i] for i in ids]))
        assert_baseline_replay(score, predictions, [original[i] for i in ids],
                               expected_domains[domain], domain)
        baselines[domain] = compact(score)
        baseline_predictions[domain], baseline_masks[domain] = predictions, masks
    results = []
    for correction in args.class_logit_corrections:
        probabilities = []
        for value in logits:
            value = value.clone()
            value[:, 1:] -= correction
            probabilities.append(span_posteriors(value))
        for anchors in (False, True):
            for cutoff in CUTOFFS:
                row_predictions = []
                for i, posterior in enumerate(probabilities):
                    selected = select_span_evidence(posterior, cutoff, anchors=[
                        (s, e, 1.) for s, e in sorted(original[i])] if anchors else ())
                    names = [RecognizerResult("KR_NAME", s, e, score)
                             for s, e, score in selected
                             if not any(s < a.end and a.start < e for a in addresses[i])]
                    row_predictions.append([*addresses[i], *names])
                evaluated = {}
                for domain, (ids, rows) in domains.items():
                    score, predictions, masks = evaluate_guard(
                        rows, guard_for(rows, [row_predictions[i] for i in ids]))
                    evaluated[domain] = dict(
                        metrics=compact(score), regression_gate=actual_mask_gate(
                            rows, baseline_predictions[domain], predictions,
                            baseline_masks[domain], masks))
                policy = dict(cutoff=cutoff, class_logit_correction=correction,
                              protect_anchors=anchors)
                results.append(dict(id=f"correction={correction}:anchors={anchors}:cutoff={cutoff}",
                                    policy=policy, sources=evaluated))
                print(json.dumps(dict(policy=policy, sources={key: dict(
                    tp=v["metrics"]["tp"], fp=v["metrics"]["fp"], fn=v["metrics"]["fn"],
                    full=v["metrics"]["fully_covered_names"], gate=v["regression_gate"]["passed"])
                    for key, v in evaluated.items()})), flush=True)
    changed = any(digest(path) != expected for path, expected in hashes.items())
    report = dict(**manifest, baselines=baselines, candidates=results,
                  baseline_replay_verified=True,
                  source_changed_during_run=changed)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if changed:
        raise ValueError("Frozen development comparison changed during execution")


if __name__ == "__main__":
    main()
