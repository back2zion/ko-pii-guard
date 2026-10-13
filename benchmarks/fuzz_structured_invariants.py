"""Deterministic JSON masking invariants on synthetic nested records.

This checks output shape, stars offsets, non-mutation and transaction failures.
It does not measure detection accuracy or real-data privacy performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard

ROOT = Path(__file__).resolve().parents[1]
KEYS = (
    "name", "firstName", "lastName", "phoneNumber", "emailAddress",
    "accountNumber", "bankName", "creditCardNumber", "address", "orderNumber",
    "note", "status", "count", "배송지", "예금주",
)
VALUES = (
    "홍길동", "김", "길동", "010-1234-5678", "821012345678",
    "hello@internal.local", "홍길동 <hello@example.com>",
    "4111111111111112", "3333120145448", "3025382178204",
    "서울특별시 강남구 테헤란로 123", "접수", "없음", "AB-123", "",
    "０１０－１２３４－５６７８", "010\u200b-1234-5678", "📎",
)
BANKS = ("농협", "카카오뱅크", "신한", "없음")


def package_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src" / "ko_pii_guard").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def random_scalar(rng: random.Random, key: str) -> object:
    if key == "bankName":
        return rng.choice(BANKS)
    if key == "count":
        return rng.randrange(10000) if rng.randrange(3) else rng.random() * 100
    if rng.randrange(8) == 0:
        return rng.randrange(10**10, 10**14)
    return rng.choice(VALUES)


def random_record(rng: random.Random) -> object:
    rows = []
    for _ in range(rng.randrange(1, 5)):
        keys = rng.sample(KEYS, rng.randrange(1, 8))
        row = {key: random_scalar(rng, key) for key in keys}
        if rng.randrange(30) == 0:
            row["010-1234-5678"] = rng.choice(VALUES)
            if rng.randrange(4) == 0:
                row["010-5678-1234"] = rng.choice(VALUES)
        if rng.randrange(40) == 0:
            row["accountNumber"] = 1234567890123.0
        if rng.randrange(5) == 0:
            row["children"] = [{"note": rng.choice(VALUES)} for _ in range(rng.randrange(1, 4))]
        rows.append(row)
    return rows[0] if len(rows) == 1 else {"records": rows}


def compare(original: object, tag: object, stars: object, partial: object) -> tuple[int, int]:
    """Return changed leaves and keys; require length-preserving star output."""
    if isinstance(original, dict):
        if (not all(isinstance(value, dict) for value in (tag, stars, partial))
                or any(len(value) != len(original) for value in (tag, stars, partial))):
            raise AssertionError("JSON object structure changed")
        changed = keys_changed = 0
        for (key, child), (tag_key, tag_child), (star_key, star_child), (
            partial_key, partial_child
        ) in zip(original.items(), tag.items(), stars.items(), partial.items(), strict=True):
            leaf_changes, nested_keys = compare(child, tag_child, star_child, partial_child)
            changed += leaf_changes
            keys_changed += nested_keys
            if not isinstance(tag_key, str) or not isinstance(star_key, str):
                raise AssertionError("JSON key became non-string")
            if len(star_key) != len(key):
                raise AssertionError("Stars changed key length")
            if any(a != b and b != "*" for a, b in zip(key, star_key, strict=True)):
                raise AssertionError("Stars changed a key to a non-star")
            if (tag_key != key) != (star_key != key):
                raise AssertionError("Tag and stars disagree on a key")
            if tag_key != key and partial_key == key:
                raise AssertionError("Partial style exposed an entire detected key")
            keys_changed += tag_key != key
        return changed, keys_changed
    if isinstance(original, list):
        if (not all(isinstance(value, list) for value in (tag, stars, partial))
                or any(len(value) != len(original) for value in (tag, stars, partial))):
            raise AssertionError("JSON list structure changed")
        sums = [compare(a, b, c, d) for a, b, c, d in zip(
            original, tag, stars, partial, strict=True
        )]
        return sum(item[0] for item in sums), sum(item[1] for item in sums)
    if original is None or type(original) in (bool, float):
        if (tag, stars, partial) != (original, original, original):
            raise AssertionError("Unsupported scalar changed")
        return 0, 0
    source = str(original)
    if not isinstance(stars, (str, int)) or len(str(stars)) != len(source):
        raise AssertionError("Stars changed value length")
    if any(a != b and b != "*" for a, b in zip(source, str(stars), strict=True)):
        raise AssertionError("Stars changed a value to a non-star")
    if (tag != original) != (stars != original):
        raise AssertionError("Tag and stars disagree on a value")
    if tag != original and partial == original:
        raise AssertionError("Partial style exposed an entire detected value")
    return int(tag != original), 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.progress_every < 1:
        parser.error("--iterations and --progress-every must be positive")
    source_sha = package_hash()
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    scorer_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rng = random.Random(args.seed)
    guard = KoreanPIIGuard(entities=SUPPORTED_ENTITIES)
    counts = Counter()
    for index in range(args.iterations):
        record = random_record(rng)
        before = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
        try:
            tag = guard.mask_json(record, style="tag")
            stars = guard.mask_json(record, style="stars")
            partial = guard.mask_json(record, style="partial")
            changes, keys = compare(record, tag, stars, partial)
            for output in (tag, stars, partial):
                json.dumps(output, ensure_ascii=False, allow_nan=False)
            counts["successful_records"] += 1
            counts["changed_leaves"] += changes
            counts["changed_keys"] += keys
        except ValueError as error:
            # A collision, explicit float PII field or invalid nested value
            # must abort without changing the caller's input.
            if not any(word in str(error) for word in (
                "duplicate JSON object keys", "floating-point", "field boundary"
            )):
                raise RuntimeError(f"Unexpected error at seed={args.seed}, index={index}") from None
            counts["expected_rejections"] += 1
        except Exception as error:
            raise RuntimeError(
                f"Structured invariant failed at seed={args.seed}, index={index}: "
                f"{type(error).__name__}"
            ) from None
        if json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) != before:
            raise AssertionError(f"Input changed at seed={args.seed}, index={index}")
        if (index + 1) % args.progress_every == 0:
            print(f"checked={index + 1}", file=sys.stderr, flush=True)
    report = {
        "scope": "synthetic_nested_json_masking_invariants_not_accuracy",
        "seed": args.seed, "iterations": args.iterations,
        "source_commit": source_commit,
        "source_sha256": source_sha,
        "scorer_sha256": scorer_sha,
        **counts,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
