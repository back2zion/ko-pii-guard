"""Extract typed-field JSON cases from the pinned Kiii² synthetic documents.

This deliberately changes the task: each annotated surface becomes the value
of an explicit schema field. It measures structured masking, not detection in
the original prose. Raw records stay outside Git; stdout has counts and hashes.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

if __package__:
    from .prepare_kiii_evaluation import (
        DATA_SHA256,
        DATASET_ID,
        REVISION,
        STRUCTURED_TYPES,
        digest_file,
        selected_rows,
    )
else:
    from prepare_kiii_evaluation import (
        DATA_SHA256,
        DATASET_ID,
        REVISION,
        STRUCTURED_TYPES,
        digest_file,
        selected_rows,
    )


def convert(row: dict, include_hard_negatives: bool = True) -> dict:
    fields = []
    targets = []
    negative_paths = []
    text = row["text"]
    for span in row["spans"]:
        category = span["category"]
        if category not in STRUCTURED_TYPES:
            continue
        start, end, surface = span["start"], span["end"], span["surface"]
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(text)
                or text[start:end] != surface):
            raise ValueError("Pinned source contains an invalid supported span")
        fields.append({category: surface})
        targets.append({"path": ["fields", len(fields) - 1, category],
                        "type": STRUCTURED_TYPES[category]})
    if include_hard_negatives:
        for span in row["hard_negatives"]:
            start, end, surface = span["start"], span["end"], span["surface"]
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(text)
                    or text[start:end] != surface):
                raise ValueError("Pinned source contains an invalid hard negative")
            fields.append({"reference": surface})
            negative_paths.append(["fields", len(fields) - 1, "reference"])
    return {"id": row["doc_id"], "record": {"fields": fields},
            "targets": targets, "negative_paths": negative_paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-rank", type=int, default=0)
    parser.add_argument("--no-hard-negatives", action="store_true")
    args = parser.parse_args()
    if args.selection_rank < 0:
        parser.error("--selection-rank must be nonnegative")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise SystemExit("Install the optional ner extra for huggingface_hub") from error
    source = Path(hf_hub_download(DATASET_ID, "data/test.jsonl", repo_type="dataset",
                                  revision=REVISION, local_files_only=True))
    if digest_file(source) != DATA_SHA256:
        raise ValueError("Pinned Kiii² source hash changed")
    rows = selected_rows(source, selection_rank=args.selection_rank)
    output = args.output.absolute()
    if (output.is_symlink() or output.is_dir()
            or (output.exists() and os.path.samefile(source, output))):
        parser.error("--output must be a distinct regular file path")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    counts = Counter()
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=f".{output.name}.", suffix=".tmp",
                                         delete=False) as temporary:
            temporary_name = temporary.name
            for row in rows:
                case = convert(row, include_hard_negatives=not args.no_hard_negatives)
                counts.update(target["type"] for target in case["targets"])
                counts["hard_negative_fields"] += len(case["negative_paths"])
                temporary.write(json.dumps(case, ensure_ascii=False) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, output)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    print(json.dumps({"dataset_revision": REVISION, "dataset_sha256": DATA_SHA256,
                      "selection_rank": args.selection_rank, "documents": len(rows),
                      "target_fields": sum(counts[kind] for kind in STRUCTURED_TYPES.values()),
                      "targets_by_entity": {
                          kind: counts[kind] for kind in STRUCTURED_TYPES.values()
                      },
                      "hard_negative_fields": counts["hard_negative_fields"],
                      "output_sha256": digest_file(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
