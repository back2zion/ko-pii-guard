"""Combine four frozen KDPII accuracy-only shards without copying source text."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from private_corpus_benchmark import write_report


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def score(tp: int, fp: int, fn: int) -> dict:
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def combine(reports: list[dict], side: str) -> dict:
    counts = ("documents", "characters", "gold_spans", "fully_masked_gold_spans",
              "negative_documents", "negative_documents_with_false_positive",
              "positive_documents", "positive_documents_with_unmasked_gold")
    combined = {key: sum(report[side][key] for report in reports) for key in counts}
    combined["total"] = score(*(sum(report[side]["total"][key] for report in reports)
                                for key in ("tp", "fp", "fn")))
    combined["full_mask_coverage"] = (
        combined["fully_masked_gold_spans"] / combined["gold_spans"]
    )
    combined["positive_document_leak_rate"] = (
        combined["positive_documents_with_unmasked_gold"] / combined["positive_documents"]
    )
    combined["negative_document_false_positive_rate"] = (
        combined["negative_documents_with_false_positive"] / combined["negative_documents"]
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--reports", nargs=4, type=Path, required=True)
    parser.add_argument("--profile", choices=("e5", "rules"), default="e5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    script_sha = digest(Path(__file__).read_bytes())
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest["valid_documents"] != 4_890 or manifest["gold_name_spans"] != 176
            or manifest["excluded_documents"] != 1 or len(manifest["shards"]) != 4):
        raise ValueError("KDPII preparation manifest does not match the frozen scope")
    full = args.input.read_bytes()
    if digest(full) != manifest["output_sha256"]:
        raise ValueError("Full KDPII evaluation input hash changed")
    parts = []
    for index, details in enumerate(manifest["shards"]):
        if details["index"] != index:
            raise ValueError("KDPII shard order changed")
        part = (args.shard_dir / f"part-{index}.jsonl").read_bytes()
        if digest(part) != details["sha256"] or part.count(b"\n") != details["documents"]:
            raise ValueError("KDPII shard contents changed")
        parts.append(part)
    if b"".join(parts) != full:
        raise ValueError("KDPII shards do not reconstruct the full input")
    report_bytes = [path.read_bytes() for path in args.reports]
    reports = [json.loads(content) for content in report_bytes]
    fixed_keys = ("baseline_commit", "current_commit", "baseline_source_sha256",
                  "current_source_sha256", "scorer_sha256", "entities", "repeats",
                  "accuracy_only", "score_threshold", "normalize_unicode", "ner_enabled",
                  "ner_model_key", "ner_score_threshold", "ner_threads", "dependencies")
    fixed = {key: reports[0][key] for key in fixed_keys}
    use_ner = args.profile == "e5"
    if (fixed["entities"] != ["KR_NAME"] or fixed["repeats"] != 0
            or not fixed["accuracy_only"] or fixed["ner_enabled"] != use_ner
            or fixed["ner_model_key"] != ("e5" if use_ner else None)
            or reports[0]["baseline"]["ner_model"] != reports[0]["current"]["ner_model"]
            or any(report["current_source_has_uncommitted_changes"] for report in reports)):
        raise ValueError("KDPII shard evaluation settings changed")
    for index, report in enumerate(reports):
        if (report["corpus_sha256"] != manifest["shards"][index]["sha256"]
                or any(report[key] != value for key, value in fixed.items())
                or report["development_corpus_sha256"] is not None):
            raise ValueError("KDPII shard provenance mismatch")
        for side in ("baseline", "current"):
            metrics = report[side]
            if (metrics["ner_model"] != reports[0][side]["ner_model"]
                    or metrics["gold_spans"] != metrics["total"]["tp"]
                    + metrics["total"]["fn"]
                    or metrics["wrong_type_same_span"]
                    or metrics["ignored_spans"] or metrics["negative_spans"]):
                raise ValueError("KDPII shard metrics cannot be combined")
    baseline, current = (combine(reports, side) for side in ("baseline", "current"))
    if (baseline["documents"] != manifest["valid_documents"]
            or baseline["gold_spans"] != manifest["gold_name_spans"]
            or current["documents"] != baseline["documents"]
            or current["gold_spans"] != baseline["gold_spans"]):
        raise ValueError("KDPII aggregate denominators changed")
    if (digest(Path(__file__).read_bytes()) != script_sha
            or args.manifest.read_bytes() != manifest_bytes
            or any(path.read_bytes() != content
                   for path, content in zip(args.reports, report_bytes, strict=True))):
        raise ValueError("KDPII aggregation inputs changed during run")
    output = {
        "scope": (f"KDPII_v1_full_test_name_only_{args.profile}_accuracy_"
                  "not_business_holdout_or_speed"),
        "source_url": manifest["source_url"],
        "source_license": manifest["source_license"],
        "source_sha256": manifest["source_sha256"],
        "preparation_script_sha256": manifest["preparation_script_sha256"],
        "converter_sha256": manifest["converter_sha256"],
        "aggregation_script_sha256": script_sha,
        "full_input_sha256": manifest["output_sha256"],
        "population": manifest["population"],
        "excluded_documents": manifest["excluded_documents"],
        "excluded_gold_name_spans": manifest["excluded_gold_name_spans"],
        "excluded_reason": manifest["excluded_reason"],
        "shards": [{"index": item["index"], "documents": item["documents"],
                    "sha256": item["sha256"]} for item in manifest["shards"]],
        "evaluation": fixed,
        "ner_model": reports[0]["current"]["ner_model"],
        "baseline": baseline,
        "current": current,
        "limitations": [
            "Includes the earlier 500-sentence sample and is not a new independent holdout.",
            "Public conversational data are not raw business documents.",
        ] + (["Possible upstream E5 training exposure is unknown."] if use_ner else []),
    }
    write_report(args.output, json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"baseline": baseline, "current": current}, ensure_ascii=False))


if __name__ == "__main__":
    main()
