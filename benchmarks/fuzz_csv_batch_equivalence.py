"""Check batched CSV masking against per-row masking on generated mixed records.

This checks behavior preservation, not PII detection accuracy or realistic data
speed. Generated values never appear in the aggregate report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import subprocess
from pathlib import Path

from private_corpus_benchmark import source_hash

from ko_pii_guard import KoreanPIIGuard
from ko_pii_guard.cli import _process_csv

ROOT = Path(__file__).resolve().parents[1]
HEADERS = ("person_name", "phone_no", "bank", "bank_account_no", "email",
           "notes", "status", "010-0000-0000")
ENTITIES = ("KR_NAME", "PHONE_NUMBER", "KR_ACCOUNT", "EMAIL_ADDRESS")


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def commit_id() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout.strip()


def generated_csv(seed: int, count: int) -> str:
    rng = random.Random(seed)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(HEADERS)
    names = ("홍길동", "이상", "Alice", "없음", "")
    phones = ("010-1234-5678", "821012345678", "없음", "")
    banks = ("농협", "카카오뱅크", "기타", "")
    accounts = ("3021234567890", "3333123456789", "없음", "")
    emails = ("hello@internal.local", "alice@example.com", "없음", "")
    notes = ("접수", "첫 줄\n둘째 줄", "010-5678-1234", "x" * 1_000)
    statuses = ("정상", "미입력", "보류", "")
    for index in range(count):
        writer.writerow((rng.choice(names), rng.choice(phones), rng.choice(banks),
                         rng.choice(accounts), rng.choice(emails), rng.choice(notes),
                         rng.choice(statuses), f"ORD-{index:05d}"))
    return output.getvalue()


def per_row_csv(content: str, guard: KoreanPIIGuard, style: str,
                strict_fields: bool) -> str:
    source = io.StringIO(content, newline="")
    target = io.StringIO(newline="")
    reader = csv.reader(source, strict=True)
    writer = csv.writer(target)
    headers = next(reader)
    masked_headers = [guard.mask(header, style=style) for header in headers]
    writer.writerow(masked_headers)
    for row in reader:
        masked = guard.mask_json(dict(zip(headers, row, strict=True)), style=style,
                                 strict_fields=strict_fields)
        writer.writerow([masked[header] for header in masked_headers])
    return target.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.rows <= 10_000 or not 1 <= args.seeds <= 100:
        parser.error("--rows must be 1..10000 and --seeds must be 1..100")
    commit = commit_id()
    package_sha = source_hash(ROOT)
    script_sha = digest(Path(__file__).read_bytes())
    comparisons = 0
    input_hashes = []
    output_hashes = []
    for seed in range(args.seeds):
        content = generated_csv(20261014 + seed, args.rows)
        input_hashes.append(digest(content.encode()))
        for style in ("tag", "stars", "partial"):
            for strict_fields in (False, True):
                batched_guard = KoreanPIIGuard(entities=ENTITIES)
                batched = io.StringIO(newline="")
                _process_csv(io.StringIO(content, newline=""), batched, batched_guard,
                             style, strict_fields)
                reference = per_row_csv(content, KoreanPIIGuard(entities=ENTITIES),
                                        style, strict_fields)
                if batched.getvalue() != reference:
                    raise RuntimeError("Batched CSV output differs from per-row reference")
                output_hashes.append(digest(reference.encode()))
                comparisons += 1
        print(f"checked_seed={seed + 1}", flush=True)
    if (commit_id() != commit or source_hash(ROOT) != package_sha
            or digest(Path(__file__).read_bytes()) != script_sha):
        raise RuntimeError("Source or equivalence checker changed during run")
    report = {
        "scope": "synthetic_csv_batch_vs_per_row_equivalence_not_detection_accuracy",
        "source_commit": commit,
        "source_sha256": package_sha,
        "checker_sha256": script_sha,
        "rows_per_seed": args.rows,
        "seeds": args.seeds,
        "styles": ["tag", "stars", "partial"],
        "strict_fields": [False, True],
        "comparisons": comparisons,
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
        "all_outputs_identical_to_per_row": True,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
