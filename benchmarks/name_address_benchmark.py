"""Exact-span and actual masking evaluation for synthetic Korean names/addresses.

This curated development corpus is not an independent real-world estimate.
Run: uv run python benchmarks/name_address_benchmark.py --check
NER: uv run --extra ner python benchmarks/name_address_benchmark.py --ner --output report.json
The optional model must already be downloaded; this evaluator loads it offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import ko_pii_guard
from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard

DEFAULT_DATA = Path(__file__).parent / "data" / "name_address.jsonl"
ENTITIES = ("KR_NAME", "KR_ADDRESS")
COUNT_KEYS = (
    "sentences", "true_positive", "false_positive", "false_negative", "sensitive_spans",
    "fully_covered_spans", "negative_sentences", "false_positive_sentences",
)


def load_cases(path: Path = DEFAULT_DATA) -> list[dict]:
    """Reject malformed annotations instead of silently changing their meaning."""
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    ids = set()
    for case in cases:
        if case["id"] in ids:
            raise ValueError(f"Duplicate case ID: {case['id']}")
        ids.add(case["id"])
        if not isinstance(case["text"], str) or not case["track"]:
            raise ValueError(f"Invalid text/track in {case['id']}")
        spans = set()
        for entity in case["expected"]:
            span = (entity["entity"], entity["start"], entity["end"])
            if (span[0] not in ENTITIES or not isinstance(span[1], int)
                    or not isinstance(span[2], int)
                    or not 0 <= span[1] < span[2] <= len(case["text"]) or span in spans):
                raise ValueError(f"Invalid or duplicate annotation in {case['id']}")
            spans.add(span)
    if not cases:
        raise ValueError("Corpus must contain at least one case")
    return cases


def _record(counter: Counter, expected: set, predicted: set, covered: set) -> None:
    counter["sentences"] += 1
    counter["true_positive"] += len(expected & predicted)
    counter["false_positive"] += len(predicted - expected)
    counter["false_negative"] += len(expected - predicted)
    counter["sensitive_spans"] += len(expected)
    counter["fully_covered_spans"] += len(expected & covered)
    if not expected:
        counter["negative_sentences"] += 1
        counter["false_positive_sentences"] += bool(predicted)


def _metrics(counter: Counter) -> dict:
    counts = {key: counter[key] for key in COUNT_KEYS}
    tp, fp, fn = (counts[key] for key in ("true_positive", "false_positive", "false_negative"))
    return {
        "counts": counts,
        "exact_span_precision": tp / (tp + fp) if tp + fp else 0.0,
        "exact_span_recall": tp / (tp + fn) if tp + fn else 0.0,
        "exact_span_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "full_sensitive_span_coverage": (
            counts["fully_covered_spans"] / counts["sensitive_spans"]
            if counts["sensitive_spans"] else 0.0
        ),
        "negative_sentence_false_positive_rate": (
            counts["false_positive_sentences"] / counts["negative_sentences"]
            if counts["negative_sentences"] else 0.0
        ),
    }


def evaluate(cases: list[dict], guard: KoreanPIIGuard) -> dict:
    """Evaluate unmodified gold spans; errors contain case IDs, never raw text."""
    total: Counter = Counter()
    by_entity = {entity: Counter() for entity in ENTITIES}
    tracks = sorted({case["track"] for case in cases})
    by_track = {track: Counter() for track in tracks}
    track_entities = {track: {entity: Counter() for entity in ENTITIES} for track in tracks}
    errors = {"exact_span": [], "uncovered_sensitive_span": [], "negative_false_positive": []}
    started = time.perf_counter()
    for case in cases:
        text, track = case["text"], case["track"]
        expected = {(e["entity"], e["start"], e["end"]) for e in case["expected"]}
        predicted = {(f.entity, f.start, f.end) for f in guard.analyze(text)}
        # Exercise the public masking API separately. Label correctness and
        # complete redaction are distinct: a wrong label can still cover a span.
        masked = guard.mask(text, style="stars")
        covered = {span for span in expected if len(masked) == len(text)
                   and masked[span[1]:span[2]] == "*" * (span[2] - span[1])}
        _record(total, expected, predicted, covered)
        _record(by_track[track], expected, predicted, covered)
        for entity in ENTITIES:
            gold = {span for span in expected if span[0] == entity}
            found = {span for span in predicted if span[0] == entity}
            _record(by_entity[entity], gold, found, covered)
            _record(track_entities[track][entity], gold, found, covered)
        if expected != predicted:
            errors["exact_span"].append(case["id"])
        if expected - covered:
            errors["uncovered_sensitive_span"].append(case["id"])
        if not expected and predicted:
            errors["negative_false_positive"].append(case["id"])
    return {
        **_metrics(total),
        "per_entity": {entity: _metrics(counts) for entity, counts in by_entity.items()},
        "per_track": {
            track: {**_metrics(by_track[track]), "per_entity": {
                entity: _metrics(counts) for entity, counts in track_entities[track].items()
            }} for track in tracks
        },
        "seconds_analyze_and_mask": time.perf_counter() - started,
        "error_ids": errors,
    }


def _source_metadata() -> dict:
    source_dir = Path(ko_pii_guard.__file__).resolve().parent
    files = {str(path.relative_to(source_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(source_dir.rglob("*.py"))}
    manifest = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"source_sha256": hashlib.sha256(manifest).hexdigest(), "source_files_sha256": files}


def _dependency_versions() -> dict:
    versions = {}
    for package in ("ko-pii-guard", "presidio-analyzer", "presidio-anonymizer", "regex",
                    "phonenumbers", "tldextract", "torch", "transformers", "huggingface-hub",
                    "tokenizers", "safetensors"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def numeric_negative_probe(guard: KoreanPIIGuard) -> dict:
    """Reuse exactly the numeric benchmark's negative corpus and RNG state."""
    from synthetic_benchmark import NEGATIVE_TEMPLATES, POSITIVE_CASES, N

    rng = random.Random(2026)
    for cases in POSITIVE_CASES.values():
        for _, generate in cases:
            for _ in range(N):
                generate(rng)
    texts = [generate(rng) for generate in NEGATIVE_TEMPLATES for _ in range(N)]
    false_positives = []
    for i, text in enumerate(texts):
        findings = guard.analyze(text)
        if findings:
            false_positives.append({"id": i, "entities": [f.entity for f in findings]})
    return {
        "sentences": len(texts), "false_positive_sentences": len(false_positives),
        "false_positive_findings": sum(len(row["entities"]) for row in false_positives),
        "seed": 2026, "samples_per_template": N,
        "data_sha256": hashlib.sha256(
            json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
        "errors": false_positives,
        "entities": guard.entities,
        "scope": "numeric synthetic negatives, not prose name accuracy",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ner", action="store_true",
                        help="Also evaluate the pinned local NER model")
    parser.add_argument("--name-context-path", type=Path,
                        help="Opt-in trained character name checkpoint (requires --ner)")
    parser.add_argument("--ner-threshold", type=float, default=0.9)
    parser.add_argument("--ner-model", choices=("e5", "kcelectra"), default="e5")
    parser.add_argument("--numeric-negatives", action="store_true",
                        help="Also screen the existing 2,200 numeric negative sentences")
    parser.add_argument("--check-negatives", action="store_true",
                        help="Fail if any evaluated profile flags a negative sentence")
    parser.add_argument("--check", action="store_true",
                        help="Require perfect rules-only structured results; prose is diagnostic")
    args = parser.parse_args()
    if args.name_context_path is not None and not args.ner:
        parser.error("--name-context-path requires --ner")
    if not 0 <= args.ner_threshold <= 1:
        parser.error("--ner-threshold must be between 0 and 1")
    cases = load_cases(args.data)
    if args.check and not any(case["track"] == "structured" for case in cases):
        parser.error("--check requires a structured track")

    source = _source_metadata()
    rules = KoreanPIIGuard(entities=SUPPORTED_ENTITIES)
    report = {
        "corpus": "curated synthetic development regression; not independent evaluation",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": ["uv", "run", "python", *sys.argv],
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **source,
        "python": platform.python_version(), "platform": platform.platform(),
        "dependency_versions": _dependency_versions(),
        "entities": rules.entities, "guard_score_threshold": rules.score_threshold,
        "coverage_definition": "actual stars masking covers every character of each gold span",
        "per_entity_negative_definition": "sentences without gold spans of that entity type",
        "timing_definition": "wall time for analyze and mask; excludes model loading",
        "models": {"name_address_rules": None},
        "results": {"name_address_rules": evaluate(cases, rules)},
    }
    ambiguous = []
    if args.data.resolve() == DEFAULT_DATA.resolve():
        ambiguity_path = DEFAULT_DATA.with_name("ambiguous_name_context.jsonl")
        ambiguous = [json.loads(line) for line in ambiguity_path.read_text().splitlines()]
        report["ambiguous_cases"] = {
            "scored": False,
            "reason": "filename alone cannot establish whether a real person is referenced",
            "data_sha256": hashlib.sha256(ambiguity_path.read_bytes()).hexdigest(),
            "predictions": {"name_address_rules": [
                {"id": c["id"], "spans": [[f.entity, f.start, f.end]
                                         for f in rules.analyze(c["text"])]} for c in ambiguous
            ]},
        }
    if args.numeric_negatives:
        report["numeric_negative_probe"] = {"default": numeric_negative_probe(KoreanPIIGuard())}
    if args.ner:
        import torch

        from ko_pii_guard.ner import MODELS, KoreanNER

        # Evaluation reproducibility only; the library does not change global
        # application thread settings. Load once and share across all cases.
        torch.set_num_threads(2)
        loaded = time.perf_counter()
        ner = KoreanNER.from_pretrained(model=args.ner_model,
                                       score_threshold=args.ner_threshold, device="cpu",
                                       name_context_path=args.name_context_path)
        model_id, revision = MODELS[args.ner_model]
        report["models"]["name_address_ner"] = {
            "model_id": model_id, "revision": revision,
            "score_threshold": ner.score_threshold, "device": ner.device,
            "max_length": ner.max_length, "stride": ner.stride,
            "batch_size": ner.batch_size, "torch_threads": torch.get_num_threads(),
            "local_files_only": True, "load_seconds": time.perf_counter() - loaded,
        }
        if args.name_context_path is not None:
            report["models"]["name_address_ner"]["name_context_checkpoint"] = {
                filename: hashlib.sha256(
                    (args.name_context_path / filename).read_bytes()
                ).hexdigest()
                for filename in ("name_context_config.json", "name_context.safetensors")
            }
        contextual = KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=ner)
        report["results"]["name_address_ner"] = evaluate(cases, contextual)
        if ambiguous:
            report["ambiguous_cases"]["predictions"]["name_address_ner"] = [
                {"id": c["id"], "spans": [[f.entity, f.start, f.end]
                                         for f in contextual.analyze(c["text"])]} for c in ambiguous
            ]
        if args.numeric_negatives:
            report["numeric_negative_probe"]["name_address_ner"] = numeric_negative_probe(
                KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=ner)
            )
    report["source_changed_during_run"] = _source_metadata() != source
    structured = report["results"]["name_address_rules"]["per_track"].get("structured")
    report["structured_rules_check_passed"] = bool(structured) and (
        structured["counts"]["false_positive"] == structured["counts"]["false_negative"] == 0
        and structured["counts"]["fully_covered_spans"] == structured["counts"]["sensitive_spans"]
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if args.check and (not report["structured_rules_check_passed"]
                       or report["source_changed_during_run"]):
        raise SystemExit(1)
    if args.check_negatives and (report["source_changed_during_run"] or any(
        result["counts"]["false_positive_sentences"] for result in report["results"].values()
    )):
        raise SystemExit("Negative-sentence regression")


if __name__ == "__main__":
    main()
