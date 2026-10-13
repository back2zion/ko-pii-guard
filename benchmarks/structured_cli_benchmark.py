"""Time default and strict JSON CLI masking on deterministic synthetic rows.

This measures end-to-end process startup, parsing, masking and file output.
It checks functional output but is not an accuracy benchmark or real workload.
Run on Linux with the environment and commit recorded in the report.
"""

from __future__ import annotations

import argparse
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


def commit_id() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout.strip()


def write_input(path: Path, records: int) -> str:
    rows = [
        {"고객명": "홍길동", "휴대폰번호": f"010-{index:04d}-1234",
         "이메일주소": "hello@internal.local", "주문번호": f"ORD-{index:05d}"}
        for index in range(records)
    ]
    path.write_text(json.dumps({"records": rows}, ensure_ascii=False), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_output(path: Path, records: int) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))["records"]
    if len(data) != records or any(
        row != {"고객명": "<KR_NAME>", "휴대폰번호": "<PHONE_NUMBER>",
                "이메일주소": "<EMAIL_ADDRESS>", "주문번호": f"ORD-{index:05d}"}
        for index, row in enumerate(data)
    ):
        raise RuntimeError("CLI output failed field-level validation")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.records < 1 or args.records > 10_000 or args.pairs < 1:
        parser.error("--records must be 1..10000 and --pairs must be positive")
    source_commit, package_sha = commit_id(), source_hash(ROOT)
    scorer_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    samples = []
    with tempfile.TemporaryDirectory(prefix="ko-pii-cli-bench-") as directory:
        base = Path(directory)
        input_path = base / "input.json"
        input_sha = write_input(input_path, args.records)
        for pair in range(args.pairs):
            for mode in (("default", "strict") if pair % 2 == 0
                         else ("strict", "default")):
                output_path = base / f"output-{len(samples)}.json"
                command = [sys.executable, "-m", "ko_pii_guard", "mask", str(input_path),
                           "--format", "json", "--entity", "KR_NAME", "--entity",
                           "PHONE_NUMBER", "--entity", "EMAIL_ADDRESS",
                           "-o", str(output_path)]
                if mode == "strict":
                    command.append("--strict-fields")
                start = time.perf_counter()
                completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                elapsed = (time.perf_counter() - start) * 1000
                if completed.returncode:
                    raise RuntimeError("CLI benchmark masking failed")
                samples.append({"mode": mode, "milliseconds": round(elapsed, 3),
                                "output_sha256": validate_output(output_path, args.records)})
    if (commit_id() != source_commit or source_hash(ROOT) != package_sha
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != scorer_sha):
        raise RuntimeError("Source or timing scorer changed during CLI timing")
    report = {
        "scope": "synthetic_json_cli_end_to_end_speed_not_detection_accuracy",
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
