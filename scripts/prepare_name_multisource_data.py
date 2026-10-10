"""Prepare original KDPII train/valid and existing KLUE training supervision.

No test corpus is opened. Optional exclusions contain text hashes only. KDPII
train IDs are sampled before reading annotations; official valid stays separate.
All outputs are immutable and raw data remains outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from fetch_kdpii_evaluation import (
    API_URL,
    RECORD_ID,
    convert_record,
    digest,
    download_or_read,
    write_immutable,
)

from ko_pii_guard.normalization import normalize_text

FILES = {
    "train.json": {"bytes": 49798233, "md5": "fa633403055e4e5918b92b2d11726434",
                   "sha256": "3c65f8c7e48196110d14fa58f591f4a0cb54682a5ab26616292ea54ec4bd46fb"},
    "valid.json": {"bytes": 6260192, "md5": "6f89923f873c840ca773d255d058150c",
                   "sha256": "3ffa97a75c584161f1096af6d2bbd252aa5e5ed556bdbe8bcb1467a57bc397c1"},
}
SALT = "ko-pii-name-multisource-v2:"


def verify_metadata(metadata):
    if (
        metadata.get("id") != RECORD_ID
        or metadata.get("metadata", {}).get("license", {}).get("id") != "cc-by-4.0"
    ):
        raise ValueError("Official KDPII record ID/license does not match")
    for name, spec in FILES.items():
        found = [row for row in metadata.get("files", []) if row.get("key") == name]
        if (
            len(found) != 1
            or found[0].get("checksum") != "md5:" + spec["md5"]
            or found[0].get("size") != spec["bytes"]
        ):
            raise ValueError(f"Official KDPII metadata differs for {name}")


def verify_source(name, content):
    spec = FILES[name]
    if (len(content) != spec["bytes"] or hashlib.md5(content).hexdigest() != spec["md5"]
            or digest(content) != spec["sha256"]):
        raise ValueError(f"Official KDPII checksum/size differs for {name}")


def select_training_ids(records, maximum=10000):
    ids = [row["sent_idx"] for row in records]
    if (
        not all(isinstance(value, str) and value for value in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("KDPII must contain unique nonempty string IDs")
    if maximum < 0:
        raise ValueError("Maximum training rows must be nonnegative")
    ordered = sorted(ids, key=lambda value: (digest((SALT + value).encode()), value))
    return ordered[:maximum] if maximum else ordered


def convert_training_record(record, split):
    if split not in {"train", "valid"}:
        raise ValueError("Only official train and valid splits are permitted")
    case = convert_record(record)
    case.update(
        source="kdpii_v1_" + split,
        split="train" if split == "train" else "validation",
        track="real_kdpii_" + split,
        char_bio=list(record["labelling_seq"]),
        char_name_bio=["O"] * len(case["text"]),
        training_ignore_spans=[
            dict(start=entity["start"], end=entity["end"], label=entity["label"])
            for entity in case["source_annotations"]
            if entity["label"] in {"PS_NICKNAME", "PS_ID"}
        ],
    )
    for entity in case["expected"]:
        start, end = entity["start"], entity["end"]
        case["char_name_bio"][start:end] = ["B-PS"] + ["I-PS"] * (end - start - 1)
    return case


def _name_signature(case):
    return tuple(sorted((e["entity"], e["start"], e["end"]) for e in case["expected"]))


def _full_signature(case):
    return tuple((e["label"], e["start"], e["end"]) for e in case["source_annotations"])


def deduplicate(cases, *, excluded_hashes=()):
    """Quarantine conflicting texts; validation wins consistent train duplicates."""
    groups = defaultdict(list)
    exclusions = set(excluded_hashes)
    excluded, conflicting, duplicate_ids, kept = [], [], [], []
    identifiers = [(case["source"], case["id"]) for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate case ID within an input source")
    for case in cases:
        text_hash = digest(case["text"].encode())
        normalized_hash = digest(normalize_text(case["text"]).text.encode())
        if text_hash in exclusions or normalized_hash in exclusions:
            excluded.append(case["id"])
        else:
            groups[text_hash].append(case)
    for text_hash, group in sorted(groups.items()):
        annotated = [case for case in group if "source_annotations" in case]
        if len({_name_signature(case) for case in group}) > 1 or len(
            {_full_signature(case) for case in annotated}
        ) > 1:
            conflicting.append(dict(text_sha256=text_hash, ids=sorted(c["id"] for c in group)))
            continue
        group.sort(key=lambda c: (
            c["split"] == "train", c["source"] != "kdpii_v1_valid",
            c["source"] == "synthetic", c["source"], c["id"],
        ))
        kept.append(group[0])
        duplicate_ids.extend(c["id"] for c in group[1:])
    kept.sort(key=lambda c: (c["source"], c["id"]))
    return kept, dict(
        excluded_ids=sorted(excluded),
        excluded_rows=len(excluded),
        duplicate_rows_removed=len(duplicate_ids),
        duplicate_ids=sorted(duplicate_ids),
        conflicting_text_groups=conflicting,
        conflicting_rows_removed=sum(len(group["ids"]) for group in conflicting),
        cross_split_exact_text_overlap=0,
    )


def _json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def _subset(cases, maximum):
    if maximum < 0:
        raise ValueError("Maximum subset size must be nonnegative")
    ordered = sorted(cases, key=lambda c: (digest((SALT + c["id"]).encode()), c["id"]))
    return ordered[:maximum] if maximum else ordered


def _read_existing(directory):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows, inputs = {}, {str(manifest_path): digest(manifest_path.read_bytes())}
    for filename in ("train.jsonl", "validation.jsonl", "synthetic-validation.jsonl"):
        path = directory / filename
        content = path.read_bytes()
        if digest(content) != manifest["outputs"][filename]["sha256"]:
            raise ValueError(f"Existing prepared data hash changed: {filename}")
        rows[filename] = [json.loads(line) for line in content.splitlines() if line]
        if len(rows[filename]) != manifest["outputs"][filename]["rows"]:
            raise ValueError(f"Existing prepared data row count changed: {filename}")
        inputs[str(path)] = digest(content)
    return rows, inputs


def _counts(cases):
    labels = Counter(e["label"] for c in cases for e in c.get("source_annotations", []))
    return dict(
        rows=len(cases),
        positive_rows=sum(bool(c["expected"]) for c in cases),
        negative_rows=sum(not c["expected"] for c in cases),
        name_spans=sum(len(c["expected"]) for c in cases),
        alias_or_handle_rows=sum(bool(c.get("training_ignore_spans")) for c in cases),
        source_counts=dict(sorted(Counter(c["source"] for c in cases).items())),
        annotation_counts=dict(sorted(labels.items())),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/ko-pii-kdpii-v1"))
    parser.add_argument("--existing-dir", type=Path,
                        default=Path("/tmp/ko-pii-name-generalization-v1-verified"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/ko-pii-name-multisource-v2"))
    parser.add_argument("--exclude-text-hashes", type=Path)
    parser.add_argument("--max-kdpii-train", type=int, default=10000)
    parser.add_argument("--max-klue-train", type=int, default=10000)
    parser.add_argument("--max-synthetic-train", type=int, default=2000)
    args = parser.parse_args()
    if any(v < 0 for v in (args.max_kdpii_train, args.max_klue_train, args.max_synthetic_train)):
        parser.error("Maximum subset counts must be nonnegative; 0 means all")
    excluded = []
    if args.exclude_text_hashes:
        excluded = json.loads(args.exclude_text_hashes.read_text())
        if not isinstance(excluded, list) or not all(
            isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value)
            for value in excluded
        ):
            raise ValueError("Exclusions must be a JSON list of SHA256 text hashes")
    meta_path = args.source_dir / "record-metadata.json"
    metadata_bytes = download_or_read(meta_path, API_URL)
    metadata = json.loads(metadata_bytes)
    verify_metadata(metadata)
    write_immutable(meta_path, metadata_bytes)
    sources = {}
    raw = {}
    for filename, spec in FILES.items():
        url = f"https://zenodo.org/api/records/{RECORD_ID}/files/{filename}/content"
        content = download_or_read(args.source_dir / filename, url)
        verify_source(filename, content)
        write_immutable(args.source_dir / filename, content)
        raw[filename] = json.loads(content)
        if not isinstance(raw[filename], list):
            raise ValueError("KDPII source must be a list of sentence records")
        sources[filename] = dict(url=url, **spec, population=len(raw[filename]))
    chosen = select_training_ids(raw["train.json"], args.max_kdpii_train)
    valid_ids = select_training_ids(raw["valid.json"], 0)
    if set(chosen) & set(valid_ids):
        raise ValueError("Official KDPII train/valid IDs overlap")
    selection = dict(
        train_ids=chosen, valid_ids=valid_ids,
        maximum_train_rows=args.max_kdpii_train,
        method=f"First IDs ordered by SHA256('{SALT}' + sent_idx); 0 means all",
        labels_read_at_selection=False,
        official_valid_resampled=False,
        official_test_opened=False,
    )
    write_immutable(args.output / "kdpii-selection.json", _json_bytes(selection))
    wanted = set(chosen)
    kdpii, invalid = [], []
    for split in ("train", "valid"):
        for record in raw[split + ".json"]:
            if split == "train" and record["sent_idx"] not in wanted:
                continue
            try:
                kdpii.append(convert_training_record(record, split))
            except (KeyError, TypeError, ValueError) as error:
                invalid.append(dict(id=record.get("sent_idx"), split=split, reason=str(error)))
    existing, inputs = _read_existing(args.existing_dir)
    original_train = existing["train.jsonl"]
    retained = [*_subset([c for c in original_train if c["source"] != "synthetic"],
                         args.max_klue_train),
                *_subset([c for c in original_train if c["source"] == "synthetic"],
                         args.max_synthetic_train),
                *existing["validation.jsonl"], *existing["synthetic-validation.jsonl"],
                *kdpii]
    unique, cleaning = deduplicate(retained, excluded_hashes=excluded)
    outputs = {
        "train.jsonl": [c for c in unique if c["split"] == "train"],
        "kdpii-validation.jsonl": [c for c in unique if c["source"] == "kdpii_v1_valid"],
        "klue-validation.jsonl": [c for c in unique if c["split"] != "train"
                                  and c["source"] in {"wikitree", "nsmc"}],
        "synthetic-validation.jsonl": [c for c in unique if c["split"] != "train"
                                       and c["source"] == "synthetic"],
    }
    output_manifest = {}
    for filename, cases in outputs.items():
        content = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases).encode()
        write_immutable(args.output / filename, content)
        output_manifest[filename] = dict(sha256=digest(content), **_counts(cases))
    hashes = sorted({digest(c["text"].encode()) for c in outputs["train.jsonl"]})
    write_immutable(args.output / "training-text-hashes.json", _json_bytes(hashes))
    if args.exclude_text_hashes:
        inputs[str(args.exclude_text_hashes)] = digest(args.exclude_text_hashes.read_bytes())
    inputs[str(Path(__file__).resolve())] = digest(Path(__file__).read_bytes())
    normalizer_path = Path(__file__).resolve().parents[1] / "src/ko_pii_guard/normalization.py"
    inputs[str(normalizer_path)] = digest(normalizer_path.read_bytes())
    mapper_path = Path(__file__).with_name("fetch_kdpii_evaluation.py")
    inputs[str(mapper_path.resolve())] = digest(mapper_path.read_bytes())
    manifest = dict(
        version="name-multisource-v2", record_id=RECORD_ID,
        source_url=f"https://zenodo.org/records/{RECORD_ID}",
        doi="10.5281/zenodo.10968609", license="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        creators=metadata["metadata"]["creators"],
        existing_data_license="KLUE CC-BY-SA-4.0; synthetic Apache-2.0",
        source_metadata_sha256=digest(metadata_bytes), sources=sources,
        input_sha256=inputs, outputs=output_manifest,
        selected_kdpii_counts_before_cleaning=_counts(kdpii),
        quarantined_invalid_records=invalid, cleaning=cleaning,
        excluded_evaluation_text_hashes=len(set(excluded)),
        evaluation_exclusion="Exact original SHA256 OR runtime normalize_text(text).text SHA256",
        policy=dict(primary="PS_NAME -> KR_NAME", alternate="PS_NAME + PS_NICKNAME",
                    training_ignore="PS_NICKNAME and PS_ID are explicit ignore spans; "
                    "trainers must apply this mask, not silently use O targets",
                    all_original_labels_preserved=True),
        limitations="Sentence-level exact-text cleaning does not prove dialogue independence. "
        "Original conversation grouping semantics are not asserted. E5 upstream used KDPII; "
        "these are new-head validation splits, not proven encoder-unseen data.",
        test_data_opened=False, model_inference_performed=False,
    )
    write_immutable(args.output / "manifest.json", _json_bytes(manifest))
    print(json.dumps(dict(outputs=output_manifest, invalid_records=len(invalid),
                          excluded_rows=cleaning["excluded_rows"],
                          duplicate_rows_removed=cleaning["duplicate_rows_removed"],
                          conflicting_rows_removed=cleaning["conflicting_rows_removed"]),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
