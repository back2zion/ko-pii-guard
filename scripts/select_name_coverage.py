"""Select a name-coverage policy on previously inspected development data only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from diagnose_name_generalization import CachedNER, reconstruct
from evaluate_name_generalization import evaluate_guard, read_selected_cases
from name_coverage_decoding import name_probabilities
from presidio_analyzer import RecognizerResult
from train_name_span_experiment import gate

from ko_pii_guard import KoreanPIIGuard
from ko_pii_guard.normalization import normalize_text

CUTOFFS = (.01, .025, .05, .1, .2, .3, .5)


def selection_rank(score, baseline):
    if (score["fully_covered_names"] < baseline["fully_covered_names"]
            or score["f1"] < baseline["f1"]
            or score["negative_false_positive_sentences"] > baseline[
                "negative_false_positive_sentences"]
            or score["unnecessary_masked_characters"] > baseline["unnecessary_masked_characters"]):
        return None
    return score["fully_covered_names"], -score["unnecessary_masked_characters"], score["f1"]


def spans_from_probabilities(probabilities, cutoff):
    start, selected = None, []
    for i, probability in enumerate([*probabilities, -1.]):
        if probability >= cutoff and start is None:
            start = i
        elif probability < cutoff and start is not None:
            selected.append((start, i, sum(probabilities[start:i]) / (i - start)))
            start = None
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path,
                        default=Path("/tmp/ko-pii-klue-ner/klue-ner-v1.1_dev.tsv"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite prior selection")
    torch.set_num_threads(2)
    development = json.loads(args.development.read_text())
    if development.get("scope") != "development" or development["source_changed_during_run"]:
        raise ValueError("Only completed frozen development results can be used")
    payload = torch.load(args.cache, weights_only=True)
    if payload["selected_ids"] != development["selected_ids"]:
        raise ValueError("Development cache IDs differ")
    for path, expected in payload["frozen_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Cached encoder input changed: {path}")
    cases = read_selected_cases(args.source, payload["selected_ids"])
    if any(normalize_text(c["text"]).text != c["text"] for c in cases):
        raise ValueError("Replay requires identity normalization, use live API for other inputs")
    original = [json.loads(line) for line in args.spans.read_text().splitlines()]
    if [r["id"] for r in original] != payload["selected_ids"]:
        raise ValueError("Cached span IDs differ")
    probabilities = [name_probabilities(logits).tolist() for logits in payload["logits"]]
    baseline = reconstruct(cases, development["baseline"]["metrics"])
    baseline_score = development["baseline"]["metrics"]
    files = [Path(__file__), Path(__file__).with_name("name_coverage_decoding.py"),
             args.development, args.cache, args.spans]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report = dict(scope="development_posterior_coverage_selection", cutoffs=CUTOFFS,
                  selection="Feasible if full coverage and F1 >= baseline, negative FP sentences "
                  "and unnecessary masked chars <= baseline; maximize full coverage then minimize "
                  "unnecessary characters then maximize F1. Per-case regression still reported.",
                  source_sha256=hashes, baseline=baseline_score, results={}, selected=None)
    best = None
    for cutoff in CUTOFFS:
        replay = CachedNER()
        guard = KoreanPIIGuard(entities=["KR_NAME"], score_threshold=0., ner=replay)
        lookup = {}
        for case, probs, raw in zip(cases, probabilities, original, strict=True):
            addresses = [RecognizerResult(r["entity"], r["start"], r["end"], r["score"])
                         for r in raw["spans"] if r["entity"] == "KR_ADDRESS"]
            names = [RecognizerResult("KR_NAME", s, e, score)
                     for s, e, score in spans_from_probabilities(probs, cutoff)
                     if not any(s < a.end and a.start < e for a in addresses)]
            lookup[case["text"]] = [*addresses, *names]
        replay.analyze = lambda text, values=lookup: values[text]
        score, predictions, _ = evaluate_guard(cases, guard)
        rank = selection_rank(score, baseline_score)
        feasible = rank is not None
        report["results"][str(cutoff)] = dict(metrics=score, feasible=feasible,
                                             regression_gate=gate(cases, baseline, predictions))
        if feasible and (best is None or rank > best):
            best, report["selected"] = rank, cutoff
        print(cutoff, {k: v for k, v in score.items() if k not in ("errors", "by_source")},
              "feasible", feasible, flush=True)
    report["source_changed_during_run"] = any(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() != value for p, value in hashes.items())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("selected", report["selected"], flush=True)
    if report["source_changed_during_run"]:
        raise RuntimeError("Frozen decision policy inputs changed")


if __name__ == "__main__":
    main()
