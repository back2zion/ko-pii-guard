"""Compare a git revision and current source in isolated processes.

Run: uv run python benchmarks/end_to_end_benchmark.py --output benchmarks/results/end-to-end.json

Both versions use the same Python interpreter and installed dependencies. Source
snapshots are taken before measuring, imports happen in separate processes, and
timings exclude imports, guard construction and a warmup call. This measures
whole analyze/mask API calls on documents of repeated synthetic mobile numbers.
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
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def _source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src" / "ko_pii_guard").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _worker(sizes: list[int], repeats: int) -> dict:
    import ko_pii_guard
    from ko_pii_guard import KoreanPIIGuard

    guard = KoreanPIIGuard()
    rows = []
    for size in sizes:
        line = "연락처 010-1234-5678 확인\n"
        text = line * size
        expected_spans = [(i * len(line) + 4, i * len(line) + 17) for i in range(size)]
        operations = {
            "analyze": lambda text=text: guard.analyze(text),
            "mask_tag": lambda text=text: guard.mask(text),
            "mask_stars": lambda text=text: guard.mask(text, style="stars"),
        }
        for name, operation in operations.items():
            result = operation()
            if name == "analyze":
                assert [(f.start, f.end) for f in result] == expected_spans
                assert all(f.entity == "PHONE_NUMBER" for f in result)
            else:
                replacement = "<PHONE_NUMBER>" if name == "mask_tag" else "*" * 13
                assert result == (f"연락처 {replacement} 확인\n" * size)
            times = []
            for _ in range(repeats):
                start = perf_counter()
                operation()
                times.append((perf_counter() - start) * 1000)
            median = statistics.median(times)
            rows.append({
                "operation": name,
                "findings": size,
                "characters": len(text),
                "median_ms": round(median, 4),
                "sample_ms": [round(value, 4) for value in times],
            })
    return {
        "imported_from": str(Path(ko_pii_guard.__file__).resolve()),
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "presidio_analyzer": version("presidio-analyzer"),
        "presidio_anonymizer": version("presidio-anonymizer"),
        "results": rows,
    }


def _archive_revision(revision: str, destination: Path) -> None:
    archive = subprocess.run(
        ["git", "archive", "--format=tar", revision, "src/ko_pii_guard"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        # Extract regular source files only; no links, paths outside the package,
        # or permissions from the archive are needed for the comparison.
        for member in source.getmembers():
            name = Path(member.name)
            if not member.isfile() or name.suffix != ".py":
                continue
            if ".." in name.parts or name.parts[:2] != ("src", "ko_pii_guard"):
                raise ValueError(f"Unexpected archive member: {member.name}")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            content = source.extractfile(member)
            assert content is not None
            target.write_bytes(content.read())


def _measure(snapshot: Path, sizes: list[int], repeats: int) -> dict:
    environment = dict(os.environ, PYTHONPATH=str(snapshot / "src"))
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker", "--repeats", str(repeats),
        "--sizes", *map(str, sizes),
    ]
    completed = subprocess.run(
        command, cwd=snapshot, env=environment, check=True, capture_output=True, text=True,
    )
    report = json.loads(completed.stdout)
    assert Path(report["imported_from"]).is_relative_to(snapshot)
    report["source_sha256"] = _source_hash(snapshot)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="v0.2.0")
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 500, 1000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.repeats < 1:
        parser.error("sizes and repeats must be positive")
    if args.worker:
        print(json.dumps(_worker(args.sizes, args.repeats)))
        return
    baseline_commit = subprocess.run(
        ["git", "rev-parse", f"{args.baseline_ref}^{{commit}}"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    current_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    current_is_dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", "src/ko_pii_guard"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip())
    with tempfile.TemporaryDirectory(prefix="ko-pii-guard-performance-") as directory:
        temporary = Path(directory)
        baseline_root, current_root = temporary / "baseline", temporary / "current"
        _archive_revision(baseline_commit, baseline_root)
        shutil.copytree(
            ROOT / "src" / "ko_pii_guard", current_root / "src" / "ko_pii_guard",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        baseline = _measure(baseline_root, args.sizes, args.repeats)
        current = _measure(current_root, args.sizes, args.repeats)
    comparison = []
    for old, new in zip(baseline["results"], current["results"], strict=True):
        assert (old["operation"], old["findings"]) == (new["operation"], new["findings"])
        comparison.append({
            "operation": new["operation"],
            "findings": new["findings"],
            "characters": new["characters"],
            "baseline_median_ms": old["median_ms"],
            "current_median_ms": new["median_ms"],
            "speedup": round(old["median_ms"] / new["median_ms"], 2),
        })
    report = {
        "scope": "whole API calls; synthetic separated mobile numbers; warm guard",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "repeats": args.repeats,
        "baseline_ref": args.baseline_ref,
        "baseline_commit": baseline_commit,
        "current_head": current_head,
        "current_includes_uncommitted_source": current_is_dirty,
        "baseline": baseline,
        "current": current,
        "comparison": comparison,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
