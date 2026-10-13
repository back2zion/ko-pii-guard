"""Prepare a fixed, label-blind finance-style synthetic evaluation sample.

Downloads a pinned Kiii² preview snapshot to the Hugging Face cache. The raw
dataset and selected documents stay outside Git; stdout contains only counts
and SHA-256 digests. See docs/kiii-evaluation.md for the scoring scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

DATASET_ID = "nmixx-fin/kiii-kiii"
REVISION = "1aa953c730cd04dac83b496a62b52e700920273b"
DATA_SHA256 = "d57b1cb0b8b67fe48a9b45ee985fb01e63b919147a8527f8231459a68b57456d"
STRUCTURED_TYPES = {
    "rrn": "KR_RRN",
    "foreigner_reg_no": "KR_FRN",
    "business_reg_no": "KR_BRN",
    "driver_license_no": "KR_DRIVER_LICENSE",
    "passport_no": "KR_PASSPORT",
    "phone_no": "PHONE_NUMBER",
    "email": "EMAIL_ADDRESS",
    "card_no": "CREDIT_CARD",
    "bank_account_no": "KR_ACCOUNT",
}
OPTIONAL_TYPES = {"person_name": "KR_NAME", "address": "KR_ADDRESS"}
UNSUPPORTED_IDENTIFIERS = {
    "online_handle", "customer_id", "corp_reg_no", "securities_account_no",
    "contract_no", "access_credential",
}
LEVELS = ("T0", "T3")
LENGTHS = ("1k", "4k", "16k")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def selected_rows(source: Path, lengths: tuple[str, ...] = LENGTHS) -> list[dict]:
    """Choose one ID per document-type/level/length cell without reading labels."""
    best: dict[tuple[str, str, str], tuple[str, dict]] = {}
    seen_ids = set()
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identifier = row["doc_id"]
            if identifier in seen_ids:
                raise ValueError("Duplicate source document ID")
            seen_ids.add(identifier)
            key = (row["doc_type"], row["variation_level"], row["context_len_bucket"])
            if key[1] not in LEVELS or key[2] not in lengths:
                continue
            rank = hashlib.sha256(identifier.encode()).hexdigest()
            if key not in best or rank < best[key][0]:
                best[key] = rank, row
    doc_types = {key[0] for key in best}
    expected = {(kind, level, length) for kind in doc_types
                for level in LEVELS for length in lengths}
    if len(doc_types) != 10 or set(best) != expected:
        raise ValueError("Pinned source is missing a selection stratum")
    return [best[key][1] for key in sorted(best)]


def convert(row: dict, include_name_address: bool,
            only_name_address: bool = False) -> dict:
    mapped = (dict(OPTIONAL_TYPES) if only_name_address else dict(STRUCTURED_TYPES))
    if include_name_address and not only_name_address:
        mapped.update(OPTIONAL_TYPES)
    ignored_categories = set(UNSUPPORTED_IDENTIFIERS)
    if only_name_address:
        ignored_categories.update(STRUCTURED_TYPES)
    elif not include_name_address:
        ignored_categories.update(OPTIONAL_TYPES)
    text = row["text"]
    entities, ignored = [], []
    for span in row["spans"]:
        start, end, category = span["start"], span["end"], span["category"]
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(text)
                or text[start:end] != span["surface"]):
            raise ValueError("Pinned source contains an invalid span")
        if category in mapped:
            entities.append({"type": mapped[category], "start": start, "end": end})
        elif category in ignored_categories:
            ignored.append({"start": start, "end": end})
    ordered = sorted(entities, key=lambda item: (item["start"], item["end"]))
    if any(a["end"] > b["start"] for a, b in zip(ordered, ordered[1:], strict=False)):
        raise ValueError("Selected source has overlapping supported spans")
    return {"id": row["doc_id"], "text": text, "entities": entities, "ignore": ignored}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--include-name-address", action="store_true")
    profile.add_argument("--only-name-address", action="store_true")
    parser.add_argument("--length", choices=LENGTHS, action="append",
                        help="select only these length buckets; default: all three")
    args = parser.parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise SystemExit("Install the optional ner extra for huggingface_hub") from error
    source = Path(hf_hub_download(DATASET_ID, "data/test.jsonl", repo_type="dataset",
                                       revision=REVISION))
    if digest_file(source) != DATA_SHA256:
        raise ValueError("Pinned source SHA-256 changed")
    lengths = tuple(dict.fromkeys(args.length or LENGTHS))
    rows = selected_rows(source, lengths)
    output = args.output.absolute()
    if (output.is_symlink() or output.is_dir()
            or (output.exists() and os.path.samefile(source, output))):
        parser.error("--output must be a distinct regular file path")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=f".{output.name}.", suffix=".tmp",
                                         delete=False) as temporary:
            temporary_name = temporary.name
            if args.only_name_address:
                chosen_types = OPTIONAL_TYPES
            elif args.include_name_address:
                chosen_types = dict(STRUCTURED_TYPES, **OPTIONAL_TYPES)
            else:
                chosen_types = STRUCTURED_TYPES
            counts = dict.fromkeys(chosen_types.values(), 0)
            for row in rows:
                case = convert(row, args.include_name_address, args.only_name_address)
                for span in case["entities"]:
                    counts[span["type"]] += 1
                temporary.write(json.dumps(case, ensure_ascii=False) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, output)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    print(json.dumps({"dataset_revision": REVISION, "dataset_sha256": DATA_SHA256,
                      "documents": len(rows), "lengths": lengths,
                      "gold_spans_by_type": counts,
                      "output_sha256": digest_file(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
