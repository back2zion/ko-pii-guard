"""Prepare the pinned KDPII v1 test split for a full name-only diagnostic.

One source row has overlapping PII annotations and cannot pass the official
BIO/offset consistency check. Exclude exactly that row, recording only a hash
of its ID; never repair or silently resample it. This is public conversational
data, not an independent business-data holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from fetch_kdpii_evaluation import TEST_SHA256, convert_record, digest, write_immutable

EXPECTED_ROWS = 4_891
EXPECTED_VALID = 4_890
EXPECTED_GOLD = 176
INVALID_ID_SHA256 = "0aef450fafd06e297afe6eabe5c7284efb6da1066624a23c46bece6bd1545111"
INVALID_REASON = "Overlapping source PII annotations"


def prepare(source: Path) -> tuple[bytes, dict]:
    raw = source.read_bytes()
    if digest(raw) != TEST_SHA256:
        raise ValueError("Pinned KDPII test checksum mismatch")
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise ValueError("Pinned KDPII test population changed")
    identifiers = [row["sent_idx"] for row in rows]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("KDPII record ID must be a nonempty string")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("KDPII record IDs must be unique")
    cases = []
    excluded = []
    excluded_gold_names = 0
    for row in rows:
        try:
            case = convert_record(row)
        except (KeyError, TypeError, ValueError) as error:
            excluded.append((hashlib.sha256(row["sent_idx"].encode()).hexdigest(),
                             str(error)))
            excluded_gold_names += sum(entity.get("label") == "PS_NAME"
                                       for entity in row.get("PII_set", []))
            continue
        cases.append({"id": case["id"], "text": case["text"],
                      "entities": [{"type": entity["entity"], "start": entity["start"],
                                    "end": entity["end"]}
                                   for entity in case["expected"]]})
    if (len(cases) != EXPECTED_VALID or sum(len(x["entities"]) for x in cases) != EXPECTED_GOLD
            or excluded != [(INVALID_ID_SHA256, INVALID_REASON)]
            or excluded_gold_names != 1):
        raise ValueError("Pinned KDPII source validation or exclusions changed")
    content = "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases).encode()
    manifest = {
        "scope": "full_pinned_KDPII_v1_test_name_only_diagnostic_not_business_holdout",
        "source_url": "https://zenodo.org/records/10968609",
        "source_license": "CC-BY-4.0",
        "source_sha256": TEST_SHA256,
        "converter_sha256": digest(Path(__file__).with_name(
            "fetch_kdpii_evaluation.py").read_bytes()),
        "preparation_script_sha256": digest(Path(__file__).read_bytes()),
        "population": EXPECTED_ROWS,
        "valid_documents": len(cases),
        "excluded_documents": len(excluded),
        "excluded_gold_name_spans": excluded_gold_names,
        "excluded_reason": INVALID_REASON,
        "excluded_id_sha256": INVALID_ID_SHA256,
        "gold_name_spans": EXPECTED_GOLD,
        "output_sha256": digest(content),
    }
    return content, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("/tmp/ko-pii-kdpii-v1/test.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path,
                        help="write four contiguous private shards for bounded evaluation")
    args = parser.parse_args()
    content, manifest = prepare(args.source)
    write_immutable(args.output, content)
    args.output.chmod(0o600)
    if args.shard_dir:
        args.shard_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        args.shard_dir.chmod(0o700)
        lines = content.splitlines(keepends=True)
        manifest["shards"] = []
        for index in range(4):
            first = index * len(lines) // 4
            last = (index + 1) * len(lines) // 4
            part = b"".join(lines[first:last])
            path = args.shard_dir / f"part-{index}.jsonl"
            write_immutable(path, part)
            path.chmod(0o600)
            manifest["shards"].append({"index": index, "documents": last - first,
                                       "sha256": digest(part)})
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
