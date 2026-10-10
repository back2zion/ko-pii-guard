"""Trace an already inspected corpus; this is diagnosis, never a new holdout score.

Offline, single-process instrumentation of the actual public inference path.
Reports IDs and coordinates, not document text. No weights or thresholds change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from name_address_benchmark import load_cases

from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard, name_context
from ko_pii_guard import ner as ner_module

ROOT = Path(__file__).resolve().parents[1]


def spans(results):
    return {(r.entity_type, r.start, r.end) for r in results if r.entity_type == "KR_NAME"}


def missing_stage(gold, raw, thresholded, filtered, merged):
    """First stage at which no head offers the exact gold span (union ceiling)."""
    for label, candidates in (("no_exact_proposal", raw),
                              ("below_threshold", thresholded),
                              ("filtered_out", filtered),
                              ("overlap_blocked", merged)):
        if gold not in candidates:
            return label
    return "downstream"


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "benchmarks/data/name_context_v11.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/name-context-v11")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cases = [c for c in load_cases(args.data) if c.get("split") == "evaluation"]
    if not cases:
        raise ValueError("No evaluation rows; only diagnose an explicitly selected corpus")
    frozen = [args.data, Path(__file__), ROOT / "src/ko_pii_guard/ner.py",
              ROOT / "src/ko_pii_guard/name_context.py", *args.checkpoint.glob("*.safetensors"),
              args.checkpoint / "name_context_config.json"]
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in frozen}
    ner = ner_module.KoreanNER.from_pretrained(name_context_path=args.checkpoint)
    guard = KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=ner)
    counts, failures = Counter(), []
    decode, filter_results = ner_module._decode_tokens, name_context.filter_name_results
    for case in cases:
        raw, thresholded, filtered = set(), set(), set()
        decode_calls = 0

        def trace_decode(tokens, threshold, raw=raw, thresholded=thresholded):
            nonlocal decode_calls
            result = decode(tokens, threshold)
            # First call decodes the backbone; its name predictions are replaced.
            if decode_calls:
                raw.update(spans(decode(tokens, 0)))
                thresholded.update(spans(result))
            decode_calls += 1
            return result

        def trace_filter(*values, filtered=filtered, **kwargs):
            result = filter_results(*values, **kwargs)
            filtered.update(spans(result))
            return result

        merged = set()
        analyze = ner.analyze

        def trace_analyze(text, analyze=analyze, merged=merged):
            result = analyze(text)
            merged.update(spans(result))
            return result

        with patch.object(ner_module, "_decode_tokens", trace_decode), \
                patch.object(name_context, "filter_name_results", trace_filter), \
                patch.object(ner, "analyze", trace_analyze):
            predicted = {(r.entity, r.start, r.end) for r in guard.analyze(case["text"])
                         if r.entity == "KR_NAME"}
        if decode_calls != 2 + len(name_context.rescue_decoders(ner.name_context_head)):
            raise ValueError("This diagnostic requires all heads on single-window cases")
        gold = {(r["entity"], r["start"], r["end"]) for r in case["expected"]
                if r["entity"] == "KR_NAME"}
        counts["sentences"] += 1
        counts["gold"] += len(gold)
        counts["true_positive"] += len(gold & predicted)
        counts["false_negative"] += len(gold - predicted)
        counts["false_positive"] += len(predicted - gold)
        counts["raw_exact_proposal_coverage"] += len(gold & raw)
        counts["threshold_exact_proposal_coverage"] += len(gold & thresholded)
        counts["filtered_exact_proposal_coverage"] += len(gold & filtered)
        for span in sorted(gold - predicted):
            reason = missing_stage(span, raw, thresholded, filtered, merged)
            counts["fn_" + reason] += 1
            failures.append(dict(id=case["id"], kind="false_negative", span=span, stage=reason))
        for span in sorted(predicted - gold):
            overlap = any(span[1] < g[2] and g[1] < span[2] for g in gold)
            reason = "overlapping_gold_boundary" if overlap else "disjoint_from_gold"
            counts["fp_" + reason] += 1
            failures.append(dict(id=case["id"], kind="false_positive", span=span, stage=reason))
    changed = any(digest(ROOT / path) != value for path, value in hashes.items())
    report = dict(scope="previously_inspected_synthetic_corpus_diagnosis",
                  not_independent_generalization_evidence=True,
                  methodology="Exact span union across heads; first unavailable stage. "
                  "Boundary FP classification is coordinate overlap, not a semantic judgment.",
                  sha256=hashes, source_changed_during_run=changed,
                  counts=dict(sorted(counts.items())), failures=failures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["counts"], ensure_ascii=False))
    if changed:
        raise SystemExit("Inputs changed during diagnosis")


if __name__ == "__main__":
    main()
