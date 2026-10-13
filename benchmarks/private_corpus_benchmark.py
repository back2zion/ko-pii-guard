"""Compare a baseline and the working tree on a local, annotated JSONL corpus.

The input stays local. Reports contain aggregate counts and timings only.
See docs/private-evaluation.md for the input contract and interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def reject_nonfinite_number(value: str) -> None:
    raise ValueError("non-finite JSON number")


def read_cases(path: Path, allowed: set[str]) -> list[dict]:
    cases = []
    seen = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line, object_pairs_hook=reject_duplicate_keys,
                                  parse_constant=reject_nonfinite_number)
                identifier, value, entities = case["id"], case["text"], case["entities"]
                if not isinstance(identifier, str) or not identifier or identifier in seen:
                    raise ValueError("id must be a unique nonempty string")
                if not isinstance(value, str) or not value:
                    raise ValueError("text must be a nonempty string")
                if not isinstance(entities, list):
                    raise ValueError("entities must be a list")
                ignored = case.get("ignore", [])
                if not isinstance(ignored, list):
                    raise ValueError("ignore must be a list")
                spans = set()
                for item in entities:
                    if not isinstance(item, dict):
                        raise ValueError("entity annotation must be an object")
                    kind = item["type"]
                    surface = item.get("surface")
                    if surface is not None and (not isinstance(surface, str) or not surface):
                        raise ValueError("surface must be a nonempty string")
                    has_start, has_end = "start" in item, "end" in item
                    if has_start != has_end:
                        raise ValueError("start and end must be supplied together")
                    if not has_start:
                        if surface is None:
                            raise ValueError("entity needs offsets or a unique surface")
                        start = value.find(surface)
                        if start < 0 or value.find(surface, start + 1) >= 0:
                            raise ValueError("surface is absent or repeated; supply offsets")
                        end = start + len(surface)
                    else:
                        start, end = item["start"], item["end"]
                    if (kind not in allowed or type(start) is not int or type(end) is not int
                            or not 0 <= start < end <= len(value)):
                        raise ValueError("invalid entity type or character offsets")
                    if surface is not None and value[start:end] != surface:
                        raise ValueError("surface does not match character offsets")
                    spans.add((kind, start, end))
                if len(spans) != len(entities):
                    raise ValueError("duplicate entity span")
                ordered = sorted(spans, key=lambda item: (item[1], item[2]))
                if any(left[2] > right[1] for left, right in zip(ordered, ordered[1:],
                                                                 strict=False)):
                    raise ValueError("overlapping gold spans")
                ignored_spans = set()
                for item in ignored:
                    if not isinstance(item, dict):
                        raise ValueError("ignored annotation must be an object")
                    start, end = item["start"], item["end"]
                    if (type(start) is not int or type(end) is not int
                            or not 0 <= start < end <= len(value)):
                        raise ValueError("invalid ignored character offsets")
                    if "surface" in item and value[start:end] != item["surface"]:
                        raise ValueError("ignored surface does not match offsets")
                    ignored_spans.add((start, end))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid input at line {line_number}: {error}") from None
            seen.add(identifier)
            cases.append({"id": identifier, "text": value, "gold": spans,
                          "ignore": ignored_spans})
    if not cases:
        raise ValueError("Input contains no documents")
    return cases


def check_split_overlap(holdout: list[dict], development: list[dict]) -> None:
    """Reject reused IDs or identical normalized documents across splits."""
    from ko_pii_guard.normalization import normalize_text

    def fingerprints(cases: list[dict]) -> set[str]:
        return {
            hashlib.sha256(normalize_text(case["text"]).text.encode("utf-8")).hexdigest()
            for case in cases
        }

    if ({case["id"] for case in holdout} & {case["id"] for case in development}
            or fingerprints(holdout) & fingerprints(development)):
        raise ValueError("Development and holdout inputs share an ID or normalized document")


def metrics(counts: Counter) -> dict:
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def covered_by_predictions(start: int, end: int, predicted: set[tuple[str, int, int]]) -> bool:
    """Require detected spans to cover each original character, not preexisting stars."""
    cursor = start
    for _, left, right in sorted(predicted, key=lambda item: (item[1], item[2])):
        if right <= cursor:
            continue
        if left > cursor:
            return False
        cursor = max(cursor, right)
        if cursor >= end:
            return True
    return False


def is_ignored_prediction(prediction: tuple[str, int, int],
                          gold: set[tuple[str, int, int]],
                          ignored: set[tuple[int, int]]) -> bool:
    """Ignore only out-of-scope detections wholly inside an annotated region."""
    _, start, end = prediction
    return (prediction not in gold
            and not any(start < right and left < end for _, left, right in gold)
            and any(left <= start and end <= right for left, right in ignored))


def worker(path: Path, entities: list[str], repeats: int, threshold: float,
           normalize_unicode: bool, use_ner: bool, ner_threshold: float,
           ner_threads: int) -> dict:
    import ko_pii_guard
    from ko_pii_guard import KoreanPIIGuard

    cases = read_cases(path, set(entities))
    ner = None
    model = None
    if use_ner:
        import torch

        from ko_pii_guard.ner import MODEL_ID, MODEL_REVISION, KoreanNER

        torch.set_num_threads(ner_threads)
        ner = KoreanNER.from_pretrained(score_threshold=ner_threshold)
        model = {"id": MODEL_ID, "revision": MODEL_REVISION}
    guard = KoreanPIIGuard(entities=entities, score_threshold=threshold,
                           normalize_unicode=normalize_unicode, ner=ner)
    counts = {kind: Counter() for kind in entities}
    coverage = {kind: Counter() for kind in entities}
    covered = 0
    gold_total = 0
    wrong_type = Counter()
    negative_docs = 0
    negative_docs_with_fp = 0
    ignored_predictions = 0
    for case in cases:
        value, gold = case["text"], case["gold"]
        findings = guard.analyze(value)
        all_predicted = {(item.entity, item.start, item.end) for item in findings}
        predicted = {
            item for item in all_predicted
            if not is_ignored_prediction(item, gold, case["ignore"])
        }
        ignored_predictions += len(all_predicted) - len(predicted)
        masked = guard.mask(value, style="stars")
        if len(masked) != len(value):
            raise RuntimeError("Stars masking changed document length")
        for key, spans in (("tp", gold & predicted), ("fp", predicted - gold),
                           ("fn", gold - predicted)):
            for kind, _, _ in spans:
                counts[kind][key] += 1
        predicted_by_span = {(start, end): kind for kind, start, end in predicted}
        for kind, start, end in gold - predicted:
            other = predicted_by_span.get((start, end))
            if other:
                wrong_type[(kind, other)] += 1
        gold_total += len(gold)
        for kind, start, end in gold:
            complete = (covered_by_predictions(start, end, predicted)
                        and masked[start:end] == "*" * (end - start))
            covered += complete
            coverage[kind]["gold"] += 1
            coverage[kind]["fully_masked"] += complete
        if not gold:
            negative_docs += 1
            negative_docs_with_fp += bool(predicted)
    analyze_times = []
    mask_times = []
    if repeats:
        guard.analyze(cases[0]["text"])
        guard.mask(cases[0]["text"], style="stars")
    for _ in range(repeats):
        started = perf_counter()
        for case in cases:
            guard.analyze(case["text"])
        analyze_times.append((perf_counter() - started) * 1000)
        started = perf_counter()
        for case in cases:
            guard.mask(case["text"], style="stars")
        mask_times.append((perf_counter() - started) * 1000)
    total = sum(counts.values(), Counter())
    return {
        "loaded_source": str(Path(ko_pii_guard.__file__).resolve()),
        "ner_model": model,
        "documents": len(cases), "characters": sum(len(c["text"]) for c in cases),
        "gold_spans": gold_total, "total": metrics(total),
        "per_entity": {kind: metrics(counts[kind]) for kind in entities},
        "fully_masked_gold_spans": covered,
        "full_mask_coverage": covered / gold_total if gold_total else None,
        "wrong_type_same_span": [
            {"gold": gold_kind, "predicted": predicted_kind, "count": count}
            for (gold_kind, predicted_kind), count in sorted(wrong_type.items())
        ],
        "mask_coverage_by_entity": {
            kind: {
                "gold": coverage[kind]["gold"],
                "fully_masked": coverage[kind]["fully_masked"],
                "rate": (coverage[kind]["fully_masked"] / coverage[kind]["gold"]
                         if coverage[kind]["gold"] else None),
            }
            for kind in entities
        },
        "negative_documents": negative_docs,
        "negative_documents_with_false_positive": negative_docs_with_fp,
        "ignored_predictions": ignored_predictions,
        "ignored_spans": sum(len(c["ignore"]) for c in cases),
        "negative_document_false_positive_rate": (
            negative_docs_with_fp / negative_docs if negative_docs else None
        ),
        "analyze_corpus_median_ms": statistics.median(analyze_times) if repeats else None,
        "analyze_corpus_samples_ms": analyze_times,
        "mask_stars_corpus_median_ms": statistics.median(mask_times) if repeats else None,
        "mask_stars_corpus_samples_ms": mask_times,
    }


def archive_revision(revision: str, destination: Path) -> None:
    archive = subprocess.run(
        ["git", "archive", "--format=tar", revision, "src/ko_pii_guard"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source.getmembers():
            name = Path(member.name)
            if not member.isfile() or name.suffix != ".py":
                continue
            if ".." in name.parts or name.parts[:2] != ("src", "ko_pii_guard"):
                raise ValueError("Unexpected source archive member")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            content = source.extractfile(member)
            assert content is not None
            target.write_bytes(content.read())


def source_hash(snapshot: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((snapshot / "src" / "ko_pii_guard").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def freeze_input(source: Path, target: Path) -> str:
    """Create a mode-0600 immutable-for-this-run copy and return its digest."""
    digest = hashlib.sha256()
    with source.open("rb") as original:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as frozen:
            while chunk := original.read(1024 * 1024):
                frozen.write(chunk)
                digest.update(chunk)
    return digest.hexdigest()


def write_report(path: Path, rendered: str) -> None:
    """Replace a complete private aggregate report with restrictive permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(rendered)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def measure(snapshot: Path, input_path: Path, entities: list[str], repeats: int,
            threshold: float, normalize_unicode: bool, use_ner: bool,
            ner_threshold: float, ner_threads: int) -> dict:
    environment = dict(os.environ, PYTHONPATH=str(snapshot / "src"))
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--input",
               str(input_path), "--repeats", str(repeats), "--threshold", str(threshold),
               "--ner-threshold", str(ner_threshold), "--ner-threads", str(ner_threads),
               "--entities", *entities]
    if not normalize_unicode:
        command.append("--no-normalize-unicode")
    if use_ner:
        command.append("--ner")
    completed = subprocess.run(
        command, cwd=snapshot, env=environment, check=True, capture_output=True, text=True,
    )
    result = json.loads(completed.stdout)
    if not Path(result["loaded_source"]).is_relative_to(snapshot):
        raise RuntimeError("Worker imported a different source tree")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--development-input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-ref", default="HEAD")
    parser.add_argument("--entities", nargs="+", default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--accuracy-only", action="store_true",
                        help="skip warm-up and repeated timing passes")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--no-normalize-unicode", action="store_true")
    parser.add_argument("--ner", action="store_true",
                        help="load the pinned local name/address model")
    parser.add_argument("--ner-threshold", type=float, default=0.9)
    parser.add_argument("--ner-threads", type=int, default=2)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 0 or (args.repeats == 0 and not args.worker):
        parser.error("--repeats must be positive (zero is reserved for internal accuracy runs)")
    repeats = 0 if args.accuracy_only else args.repeats
    if not 0.0 <= args.threshold <= 1.0 or not 0.0 <= args.ner_threshold <= 1.0:
        parser.error("thresholds must be between 0 and 1")
    if args.ner_threads < 1:
        parser.error("--ner-threads must be positive")
    from ko_pii_guard.analyzer import DEFAULT_ENTITIES, SUPPORTED_ENTITIES

    entities = args.entities or list(DEFAULT_ENTITIES)
    if len(set(entities)) != len(entities) or not set(entities) <= set(SUPPORTED_ENTITIES):
        parser.error("--entities must be unique supported entity names")
    if args.ner and not {"KR_NAME", "KR_ADDRESS"}.intersection(entities):
        parser.error("--ner requires KR_NAME or KR_ADDRESS in --entities")
    input_path = args.input.resolve(strict=True)
    if args.output and (args.output.is_symlink() or args.output.is_dir()):
        parser.error("--output must be a regular file path, not a symlink or directory")
    output_path = args.output.absolute() if args.output else None
    development_path = (args.development_input.resolve(strict=True)
                        if args.development_input else None)
    if output_path and any(
        output_path == source or (output_path.exists() and os.path.samefile(output_path, source))
        for source in (input_path, development_path) if source
    ):
        parser.error("--output must differ from the input files")
    if args.worker:
        read_cases(input_path, set(entities))
        print(json.dumps(worker(input_path, entities, repeats, args.threshold,
                                not args.no_normalize_unicode, args.ner,
                                args.ner_threshold, args.ner_threads)))
        return
    if development_path == input_path:
        parser.error("--development-input must differ from --input")
    baseline_commit = subprocess.run(
        ["git", "rev-parse", f"{args.baseline_ref}^{{commit}}"], cwd=ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="ko-pii-private-eval-") as directory:
        base = Path(directory)
        frozen_input = base / "holdout.jsonl"
        corpus_digest = freeze_input(input_path, frozen_input)
        holdout = read_cases(frozen_input, set(entities))
        development_digest = None
        if development_path:
            frozen_development = base / "development.jsonl"
            development_digest = freeze_input(development_path, frozen_development)
            development = read_cases(frozen_development, set(entities))
            check_split_overlap(holdout, development)
        baseline_root, current_root = base / "baseline", base / "current"
        archive_revision(baseline_commit, baseline_root)
        shutil.copytree(ROOT / "src" / "ko_pii_guard", current_root / "src" / "ko_pii_guard",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        baseline_hash, current_hash = source_hash(baseline_root), source_hash(current_root)
        baseline = measure(baseline_root, frozen_input, entities, repeats,
                           args.threshold, not args.no_normalize_unicode, args.ner,
                           args.ner_threshold, args.ner_threads)
        current = measure(current_root, frozen_input, entities, repeats,
                          args.threshold, not args.no_normalize_unicode, args.ner,
                          args.ner_threshold, args.ner_threads)
    if baseline["gold_spans"] != current["gold_spans"]:
        raise RuntimeError("Workers evaluated different inputs")
    if baseline["ner_model"] != current["ner_model"]:
        raise RuntimeError("Workers used different NER model revisions")
    report = {
        "baseline_commit": baseline_commit,
        "current_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "current_source_has_uncommitted_changes": bool(subprocess.run(
            ["git", "status", "--porcelain", "--", "src/ko_pii_guard"], cwd=ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip()),
        "baseline_source_sha256": baseline_hash,
        "current_source_sha256": current_hash,
        "corpus_sha256": corpus_digest,
        "development_corpus_sha256": development_digest,
        "entities": entities, "repeats": repeats,
        "accuracy_only": args.accuracy_only,
        "score_threshold": args.threshold,
        "normalize_unicode": not args.no_normalize_unicode,
        "ner_enabled": args.ner,
        "ner_score_threshold": args.ner_threshold if args.ner else None,
        "ner_threads": args.ner_threads if args.ner else None,
        "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": {
            package: version(package)
            for package in (
                "presidio-analyzer", "presidio-anonymizer", "phonenumbers", "tldextract"
            )
        },
        "baseline": baseline, "current": current,
        "scope": "exact type+character span; stars full-span coverage; warm analyze/mask time",
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if output_path:
        write_report(output_path, rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
