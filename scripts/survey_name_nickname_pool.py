"""Survey nickname/handle supervision in the official KDPII v1 train and valid splits.

Reads only the two checksum-verified official files (not test.json and no reserve), counts
labels and structure, and writes aggregate numbers plus hashes. No sentence text is stored.
Run with an isolated interpreter:  python -I scripts/survey_name_nickname_pool.py DIR OUT
"""

from __future__ import annotations

import ast
import collections
import hashlib
import json
import sys
from pathlib import Path

EXPECTED_MD5 = {"train.json": "fa633403055e4e5918b92b2d11726434",
                "valid.json": "6f89923f873c840ca773d255d058150c"}
TARGETS = ("PS_NICKNAME", "PS_ID", "PS_NAME")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def pii(record):
    value = record["PII_set"]
    return ast.literal_eval(value) if isinstance(value, str) else value


def main(directory, output):
    directory = Path(directory)
    hash_files = Path("benchmarks/results/name-p0-ledger-hashes")
    evaluation = {}
    for name in ("kdpii_test_500", "kdpii_fresh_1992"):
        for key, value in json.loads((hash_files / f"{name}.json").read_text())[
                "id_to_text_sha256"].items():
            evaluation[value] = key
    report = dict(scope="kdpii_nickname_pool_survey", splits={})
    strings = {s: {t: collections.Counter() for t in TARGETS} for s in EXPECTED_MD5}
    for filename, md5 in EXPECTED_MD5.items():
        raw = (directory / filename).read_bytes()
        if hashlib.md5(raw).hexdigest() != md5:
            raise ValueError(f"checksum differs: {filename}")
        records = json.loads(raw)
        stats = collections.Counter()
        groups = collections.defaultdict(lambda: collections.Counter())
        label_counts = collections.Counter()
        for record in records:
            entities = pii(record)
            labels = [e["label"] for e in entities]
            label_counts.update(labels)
            group = record["sent_idx"].split("_")[0]
            stats["records"] += 1
            if sha(record["sentence"]) in evaluation:
                stats["text_identical_to_evaluation_text"] += 1
            for target in TARGETS:
                n = labels.count(target)
                if n:
                    stats[f"records_with_{target}"] += 1
                    stats[f"entities_{target}"] += n
                    groups[target][group] += n
                    for e in entities:
                        if e["label"] == target:
                            strings[filename][target][e["form"]] += 1
            if "PS_NICKNAME" in labels and "PS_NAME" in labels:
                stats["records_with_nickname_and_name"] += 1
            if "PS_NICKNAME" in labels and "PS_NAME" not in labels:
                stats["records_with_nickname_without_name"] += 1
        report["splits"][filename] = dict(
            md5=md5, sha256=sha(raw.decode("utf-8")) if False else hashlib.sha256(raw).hexdigest(),
            stats=dict(stats),
            groups_by_id_prefix={t: len(groups[t]) for t in TARGETS},
            total_prefix_groups=len({r["sent_idx"].split("_")[0] for r in records}),
            unique_strings={t: len(strings[filename][t]) for t in TARGETS},
            top_nickname_strings_count=strings[filename]["PS_NICKNAME"].most_common(5)[0][1]
            if strings[filename]["PS_NICKNAME"] else 0,
            entity_labels_total=len(label_counts))
    valid_nick = set(strings["valid.json"]["PS_NICKNAME"])
    train_nick = set(strings["train.json"]["PS_NICKNAME"])
    report["nickname_string_overlap_train_valid"] = dict(
        valid_unique=len(valid_nick), shared_with_train=len(valid_nick & train_nick))
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:3])
