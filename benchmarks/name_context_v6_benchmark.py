"""Strict per-span nonregression gate; aggregate gains cannot hide new errors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from name_address_benchmark import _source_metadata, evaluate, load_cases

from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard
from ko_pii_guard.name_context import load_name_head
from ko_pii_guard.ner import MODEL_ID, MODEL_REVISION, KoreanNER

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/data"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nonregression_gate(cases, previous, current):
    protected, recovered, introduced = set(), set(), set()
    previous_false, current_false = set(), set()
    for case in cases:
        identifier = case["id"]
        gold = {(e["entity"], e["start"], e["end"]) for e in case["expected"]}
        before, after = previous[case["text"]], current[case["text"]]
        protected.update((identifier, *span) for span in gold & before)
        recovered.update((identifier, *span) for span in gold & after)
        previous_false.update((identifier, *span) for span in before - gold)
        current_false.update((identifier, *span) for span in after - gold)
    introduced = current_false - previous_false
    lost = protected - recovered
    return dict(
        protected_correct_spans=len(protected),
        lost_correct_spans=sorted(lost), introduced_false_positive_spans=sorted(introduced),
        previous_false_positive_spans=len(previous_false),
        current_false_positive_spans=len(current_false),
        nonregression_passed=not lost and not introduced,
    )


class RecordedGuard:
    def __init__(self, guard):
        self.guard = guard
        self.predictions = {}

    def analyze(self, text):
        findings = self.guard.analyze(text)
        self.predictions[text] = {(f.entity, f.start, f.end) for f in findings}
        return findings

    def mask(self, text, **kwargs):
        return self.guard.mask(text, **kwargs)


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "evaluation"), required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "artifacts/name-context-v6-filter-retry")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    sources = _source_metadata()
    hashes = {p.name: digest(p) for p in DATA.glob("*.jsonl")}
    checkpoints = {"candidate": args.checkpoint, "v3": ROOT / "artifacts/name-context-v3"}
    checkpoint_hashes = {key: {name: digest(path / name) for name in (
        "name_context_config.json", "name_context.safetensors", "span_filter.safetensors"
    ) if (path / name).exists()} for key, path in checkpoints.items()}
    ner = KoreanNER.from_pretrained(name_context_path=args.checkpoint)
    report = dict(scope=args.split, data_sha256=hashes, source=sources,
                  benchmark_sha256=digest(Path(__file__)), checkpoint_sha256=checkpoint_hashes)
    if args.split == "development":
        corpora = {name: load_cases(DATA / name) for name in (
            "name_context_v4_regressions.jsonl", "name_context_v3_regressions.jsonl",
            "name_context_v2_regressions.jsonl", "name_context.jsonl", "name_address.jsonl",
            "business_korean.jsonl", "name_field_boundaries.jsonl",
        )}
        corpora["v1_retired_evaluation"] = [c for c in load_cases(
            DATA / "name_context_training.jsonl"
        ) if c["split"] == "evaluation"]
        guard = KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=ner)
        results = {}
        for name, cases in corpora.items():
            results[name] = evaluate(cases, guard)
            print(name, results[name]["counts"], flush=True)
        report["results"] = results
        report["acceptance_passed"] = all(
            r["counts"]["false_positive"] == r["counts"]["false_negative"] == 0
            and r["counts"]["fully_covered_spans"] == r["counts"]["sensitive_spans"]
            for r in results.values()
        )
    else:
        cases = [c for c in load_cases(DATA / "name_context_v6.jsonl")
                 if c["split"] == "evaluation"]
        results, predictions = {}, {}
        for profile in ("v3", "candidate"):
            ner.name_context_head = load_name_head(
                checkpoints[profile], model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu"
            )
            guard = RecordedGuard(KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=ner))
            results[profile] = evaluate(cases, guard)
            predictions[profile] = guard.predictions
            print(profile, results[profile]["counts"], flush=True)
        report["results"] = results
        report["span_gate"] = nonregression_gate(cases, predictions["v3"], predictions["candidate"])
        before, after = (results[p]["counts"] for p in ("v3", "candidate"))
        report["negative_improved"] = (
            after["false_positive_sentences"] < before["false_positive_sentences"]
            if before["false_positive_sentences"] else after["false_positive_sentences"] == 0
        )
        report["acceptance_passed"] = report["span_gate"]["nonregression_passed"] and (
            report["negative_improved"]
            and after["fully_covered_spans"] >= before["fully_covered_spans"]
        )
    report["source_changed_during_run"] = sources != _source_metadata() or any(
        digest(DATA / name) != expected for name, expected in hashes.items()
    ) or any(digest(checkpoints[key] / name) != expected
             for key, values in checkpoint_hashes.items() for name, expected in values.items())
    if report["source_changed_during_run"]:
        raise RuntimeError("Frozen inputs changed during verification")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if args.check and not report["acceptance_passed"]:
        raise SystemExit("Candidate rejected: a protected span was lost or errors did not improve")


if __name__ == "__main__":
    main()
