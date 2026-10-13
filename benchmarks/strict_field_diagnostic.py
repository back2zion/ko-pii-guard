"""Compare normal and strict structured masking on the same local annotations.

This is accuracy-only. Reports contain aggregate counts and reproducibility
hashes, never record text, IDs, paths or individual predictions. The caller
must label non-PII status values in typed fields as negative_paths to measure
strict mode's overmasking cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

from ko_pii_guard import KoreanPIIGuard
from ko_pii_guard.analyzer import DEFAULT_ENTITIES, SUPPORTED_ENTITIES

if __package__:
    from .private_corpus_benchmark import (
        archive_revision,
        freeze_input,
        source_hash,
        write_report,
    )
    from .structured_corpus_benchmark import check_split_overlap, evaluate, read_cases
else:
    from private_corpus_benchmark import archive_revision, freeze_input, source_hash, write_report
    from structured_corpus_benchmark import check_split_overlap, evaluate, read_cases

ROOT = Path(__file__).resolve().parents[1]


class StrictGuard:
    def __init__(self, guard: KoreanPIIGuard):
        self.guard = guard
        self.entities = guard.entities

    def mask_json(self, value: object, style: str = "tag") -> object:
        return self.guard.mask_json(value, style=style, strict_fields=True)


def commit_id() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout.strip()


def scorer_hash() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("structured_corpus_benchmark.py"),
                 Path(__file__).with_name("private_corpus_benchmark.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--development-input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--entities", nargs="+", default=None)
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()
    entities = args.entities or list(DEFAULT_ENTITIES)
    if len(set(entities)) != len(entities) or not set(entities) <= set(SUPPORTED_ENTITIES):
        parser.error("--entities must be unique supported entity names")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    input_path = args.input.resolve(strict=True)
    development_path = (args.development_input.resolve(strict=True)
                        if args.development_input else None)
    if development_path == input_path:
        parser.error("--development-input must differ from --input")
    output_path = args.output.absolute() if args.output else None
    if (output_path and (output_path.is_symlink() or output_path.is_dir()
                         or output_path == input_path or output_path == development_path
                         or (output_path.exists() and any(
                             os.path.samefile(output_path, source)
                             for source in (input_path, development_path) if source
                         )))):
        parser.error("--output must differ from inputs and cannot be a symlink or directory")
    source_commit = commit_id()
    package_sha = source_hash(ROOT)
    score_sha = scorer_hash()
    with tempfile.TemporaryDirectory(prefix="ko-pii-strict-eval-") as directory:
        frozen = Path(directory) / "input.jsonl"
        corpus_sha = freeze_input(input_path, frozen)
        cases = read_cases(frozen, set(entities))
        development_sha = None
        if development_path:
            frozen_development = Path(directory) / "development.jsonl"
            development_sha = freeze_input(development_path, frozen_development)
            check_split_overlap(cases, read_cases(frozen_development, set(entities)))
        guard = KoreanPIIGuard(entities=entities, score_threshold=args.threshold)
        normal = evaluate(cases, guard)
        strict = evaluate(cases, StrictGuard(guard))
    if (commit_id() != source_commit or source_hash(ROOT) != package_sha
            or scorer_hash() != score_sha):
        raise RuntimeError("Source or scorer changed during evaluation")
    report = {
        "scope": "same_source_structured_default_vs_strict_accuracy_only",
        "source_commit": source_commit, "source_sha256": package_sha,
        "scorer_sha256": score_sha, "corpus_sha256": corpus_sha,
        "development_corpus_sha256": development_sha,
        "source_has_uncommitted_changes": package_sha != _head_source_hash(source_commit),
        "entities": entities, "threshold": args.threshold,
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in (
            "presidio-analyzer", "presidio-anonymizer", "phonenumbers", "tldextract",
        )},
        "default": normal, "strict_fields": strict,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if output_path:
        write_report(output_path, rendered)
    print(rendered, end="")


def _head_source_hash(commit: str) -> str:
    """Hash the committed package in a temporary archive without touching Git state."""
    with tempfile.TemporaryDirectory(prefix="ko-pii-strict-head-") as directory:
        archive_revision(commit, Path(directory))
        return source_hash(Path(directory))


if __name__ == "__main__":
    main()
