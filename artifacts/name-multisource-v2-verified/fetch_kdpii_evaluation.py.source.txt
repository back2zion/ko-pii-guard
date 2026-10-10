"""Pin official KDPII v1 test and select records by ID before mapping labels.

PS_NAME is the primary KR_NAME target. PS_NICKNAME is kept explicitly for an
alternate broader policy; all source annotations remain available. No inference
or training occurs. Upstream E5 training exposure is unknown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

RECORD_ID = 10968609
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
TEST_URL = f"https://zenodo.org/records/{RECORD_ID}/files/test.json?download=1"
TEST_SHA256 = "d2a4141c567528c7a1255d6325c42e0988ff223e71f296d9ac4fac20b01ed7fe"
TEST_MD5 = "08eebd98593c1bad23fbf29053b3f9d2"
TEST_SIZE = 6124918
SALT = "ko-pii-kdpii-v1:"


def digest(content):
    return hashlib.sha256(content).hexdigest()


def write_immutable(path, content):
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"Refusing to overwrite different bytes: {path.name}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def download_or_read(path, url):
    if path.exists():
        return path.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": "ko-pii-guard-research"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def verify_metadata(record):
    if (
        record.get("id") != RECORD_ID
        or record.get("metadata", {}).get("license", {}).get("id") != "cc-by-4.0"
    ):
        raise ValueError("Official KDPII record ID/license does not match")
    files = [f for f in record.get("files", []) if f.get("key") == "test.json"]
    if (
        len(files) != 1
        or files[0].get("checksum") != f"md5:{TEST_MD5}"
        or (files[0].get("size") != TEST_SIZE)
    ):
        raise ValueError("Official KDPII test metadata does not match pinned checksum/size")


def select_ids(ids, count=500):
    if not ids or not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Expected nonempty, unique string IDs")
    if not 1 <= count <= len(ids):
        raise ValueError("Invalid sample size")
    return sorted(ids, key=lambda i: digest((SALT + i).encode()))[:count]


def convert_record(record):
    identifier, text = record["sent_idx"], record["sentence"]
    chars, tags = record["sent_seq"], record["labelling_seq"]
    if (
        not isinstance(identifier, str)
        or not isinstance(text, str)
        or not isinstance(chars, list)
        or chars != list(text)
        or not isinstance(tags, list)
        or len(tags) != len(text)
    ):
        raise ValueError("Sentence, character sequence and BIO dimensions disagree")
    expected_tags = ["O"] * len(text)
    annotations = []
    for entity in record["PII_set"]:
        start, end, label, form = (entity[k] for k in ("begin", "end", "label", "form"))
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(text)
            or not isinstance(label, str)
            or not label
            or not isinstance(form, str)
            or text[start:end] != form
        ):
            raise ValueError("PII span does not match its original surface")
        if any(tag != "O" for tag in expected_tags[start:end]):
            raise ValueError("Overlapping source PII annotations")
        expected_tags[start] = "B-" + label
        expected_tags[start + 1 : end] = ["I-" + label] * (end - start - 1)
        annotations.append(dict(label=label, start=start, end=end, form=form))
    if expected_tags != tags:
        raise ValueError("PII spans and original character BIO labels disagree")
    annotations.sort(key=lambda e: (e["start"], e["end"], e["label"]))
    primary = [
        dict(entity="KR_NAME", start=e["start"], end=e["end"])
        for e in annotations
        if e["label"] == "PS_NAME"
    ]
    expanded = [
        dict(entity="KR_NAME", start=e["start"], end=e["end"])
        for e in annotations
        if e["label"] in {"PS_NAME", "PS_NICKNAME"}
    ]
    return dict(
        id=identifier,
        text=text,
        expected=primary,
        alternate_expected=expanded,
        source_annotations=annotations,
        source="kdpii_v1_test",
        split="evaluation",
        track="external_dialogue_person",
        license="CC-BY-4.0",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/ko-pii-kdpii-v1"))
    parser.add_argument("--sample-size", type=int, default=500)
    args = parser.parse_args()
    metadata_path, test_path = args.output / "record-metadata.json", args.output / "test.json"
    metadata_bytes = download_or_read(metadata_path, API_URL)
    metadata = json.loads(metadata_bytes)
    verify_metadata(metadata)
    write_immutable(metadata_path, metadata_bytes)
    test_bytes = download_or_read(test_path, TEST_URL)
    if digest(test_bytes) != TEST_SHA256 or len(test_bytes) != TEST_SIZE:
        raise ValueError("Official KDPII test SHA256/size mismatch")
    write_immutable(test_path, test_bytes)
    rows = json.loads(test_bytes)
    if not isinstance(rows, list):
        raise ValueError("KDPII v1 test must be a list of sentence records")
    ids = [r["sent_idx"] for r in rows]
    chosen = select_ids(ids, args.sample_size)
    selection = dict(
        record_id=str(RECORD_ID),
        source_sha256=TEST_SHA256,
        population=len(ids),
        selection=f"first{args.sample_size} IDs by SHA256('{SALT}' + sent_idx)",
        selected_ids=chosen,
        labels_read=False,
        unselected_records_not_inspected=True,
    )
    # Freeze label-independent selection before accessing selected PII annotations.
    selection_bytes = (json.dumps(selection, indent=2) + "\n").encode()
    write_immutable(args.output / "selection.json", selection_bytes)
    selected = {r["sent_idx"]: r for r in rows if r["sent_idx"] in set(chosen)}
    cases, failures = [], []
    for identifier in chosen:
        try:
            cases.append(convert_record(selected[identifier]))
        except (KeyError, TypeError, ValueError) as error:
            failures.append(dict(id=identifier, reason=str(error)))
    validation = dict(selected_records=len(chosen), valid_records=len(cases), failures=failures)
    write_immutable(
        args.output / "validation.json", (json.dumps(validation, indent=2) + "\n").encode()
    )
    if failures:
        raise ValueError(
            f"{len(failures)} selected source records are inconsistent; no cases exported"
        )
    cases_bytes = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases).encode()
    write_immutable(args.output / "cases.jsonl", cases_bytes)
    counts = Counter(e["label"] for case in cases for e in case["source_annotations"])
    source = dict(
        record_id=RECORD_ID,
        doi="10.5281/zenodo.10968609",
        url="https://zenodo.org/records/10968609",
        version="original v1 official test split",
        license="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        creators=metadata["metadata"]["creators"],
        files={
            "test.json": dict(url=TEST_URL, sha256=TEST_SHA256, md5=TEST_MD5, bytes=TEST_SIZE),
            "record-metadata.json": dict(url=API_URL, sha256=digest(metadata_bytes)),
            "selection.json": dict(sha256=digest(selection_bytes)),
            "cases.jsonl": dict(sha256=digest(cases_bytes)),
        },
        preparation_script_sha256=digest(Path(__file__).read_bytes()),
        population=len(rows),
        selected_rows=len(cases),
        annotation_counts=dict(sorted(counts.items())),
        primary_gold_names=sum(len(c["expected"]) for c in cases),
        primary_negative_sentences=sum(not c["expected"] for c in cases),
        alternate_gold_names_and_nicknames=sum(len(c["alternate_expected"]) for c in cases),
        policy=dict(
            primary="PS_NAME -> KR_NAME",
            alternate="PS_NAME + PS_NICKNAME -> KR_NAME",
            handles="PS_ID excluded; original annotations preserved",
            nickname_false_positives="Primary exact scorer counts nickname predictions as "
            "FP; report alternate broad policy separately, never silently relabel",
        ),
        limitations="Different conversational source from KLUE. E5 publisher used KDPII, "
        "but source version and training membership are not independently known. Do not "
        "claim upstream E5-unseen. ID sampling includes negatives; no label/error selection.",
        no_training_or_inference=True,
        raw_data_redistributed_in_repository=False,
    )
    write_immutable(args.output / "source.json", (json.dumps(source, indent=2) + "\n").encode())
    print(
        json.dumps(
            {
                key: source[key]
                for key in (
                    "population",
                    "selected_rows",
                    "primary_gold_names",
                    "primary_negative_sentences",
                    "alternate_gold_names_and_nicknames",
                    "annotation_counts",
                    "policy",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
