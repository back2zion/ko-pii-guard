"""Measure deterministic CSV CLI throughput independently of detection accuracy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if __package__:
    from .private_corpus_benchmark import source_hash, write_report
else:
    from private_corpus_benchmark import source_hash, write_report

ROOT = Path(__file__).resolve().parents[1]
HEADERS = ("고객명", "휴대폰번호", "이메일주소", "주문번호")


def commit_id() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_input(path: Path, records: int) -> str:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADERS)
        writer.writerows(("홍길동", f"010-{index:04d}-1234",
                          "hello@internal.local", f"ORD-{index:05d}")
                         for index in range(records))
    return sha256(path)


def validate_output(path: Path, records: int) -> str:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if tuple(next(reader)) != HEADERS:
            raise RuntimeError("CSV output headers changed")
        count = 0
        for index, row in enumerate(reader):
            if index >= records or row != ["<KR_NAME>", "<PHONE_NUMBER>",
                                          "<EMAIL_ADDRESS>", f"ORD-{index:05d}"]:
                raise RuntimeError("CSV output failed field validation")
            count += 1
        if count != records:
            raise RuntimeError("CSV output row count changed")
    return sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.records <= 100_000 or args.pairs < 1:
        parser.error("--records must be 1..100000 and --pairs must be positive")
    source_commit, package_sha = commit_id(), source_hash(ROOT)
    scorer_sha = sha256(Path(__file__))
    samples = []
    with tempfile.TemporaryDirectory(prefix="ko-pii-csv-bench-") as directory:
        base = Path(directory)
        input_path = base / "input.csv"
        input_sha = write_input(input_path, args.records)
        for pair in range(args.pairs):
            for mode in (("default", "strict") if pair % 2 == 0
                         else ("strict", "default")):
                output_path = base / f"output-{len(samples)}.csv"
                command = [sys.executable, "-m", "ko_pii_guard", "mask", str(input_path),
                           "--format", "csv", "--entity", "KR_NAME", "--entity",
                           "PHONE_NUMBER", "--entity", "EMAIL_ADDRESS",
                           "-o", str(output_path)]
                if mode == "strict":
                    command.append("--strict-fields")
                start = time.perf_counter()
                completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                elapsed = (time.perf_counter() - start) * 1000
                if completed.returncode:
                    raise RuntimeError("CSV CLI benchmark masking failed")
                samples.append({"mode": mode, "milliseconds": round(elapsed, 3),
                                "output_sha256": validate_output(output_path, args.records)})
    if (commit_id() != source_commit or source_hash(ROOT) != package_sha
            or sha256(Path(__file__)) != scorer_sha):
        raise RuntimeError("Source or timing scorer changed during CSV timing")
    report = {
        "scope": "synthetic_csv_cli_end_to_end_speed_not_detection_accuracy",
        "source_commit": source_commit, "source_sha256": package_sha,
        "scorer_sha256": scorer_sha, "input_sha256": input_sha,
        "records": args.records, "pairs": args.pairs,
        "python": platform.python_version(), "platform": platform.platform(),
        "samples": samples,
        "median_milliseconds": {
            mode: statistics.median(sample["milliseconds"] for sample in samples
                                    if sample["mode"] == mode)
            for mode in ("default", "strict")
        },
        "outputs_identical": len({sample["output_sha256"] for sample in samples}) == 1,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        write_report(args.output, rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
