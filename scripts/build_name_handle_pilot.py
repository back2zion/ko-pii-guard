"""Build the nickname/handle review pilot from the official KDPII v1 train split.

Selection is by SHA-256 rank of the record ID (salted) BEFORE any label-based ranking is used
within a stratum, at most one record per conversation-ID prefix, and positives and negatives
never share a prefix. Evaluation-identical texts are excluded first. Source labels are kept
verbatim; the product-policy field starts as 'unreviewed' and must be filled by independent
reviewers who do not see model predictions. Run: python -I scripts/build_name_handle_pilot.py
TRAIN_JSON OUT_JSONL OUT_MANIFEST
"""

from __future__ import annotations

import ast
import collections
import hashlib
import json
import sys
from pathlib import Path

SALT = "ko-pii-name-handle-pilot-v1:"
TRAIN_MD5 = "fa633403055e4e5918b92b2d11726434"
PLAN = [  # (stratum, source labels that must appear, quota, role)
    ("nickname", {"PS_NICKNAME"}, 140, "personal"),
    ("account_id", {"PS_ID"}, 60, "personal"),
    ("club_group_name", {"OGG_CLUB"}, 120, "non_personal"),
    ("workplace_name", {"OG_WORKPLACE"}, 40, "non_personal"),
    ("school_name", {"OGG_EDUCATION"}, 40, "non_personal"),
]
FORBIDDEN_FOR_NEGATIVE = {"PS_NICKNAME", "PS_ID", "PS_NAME"}
TARGET_LABELS = {"personal": {"PS_NICKNAME", "PS_ID"},
                 "non_personal": {"OGG_CLUB", "OG_WORKPLACE", "OGG_EDUCATION"}}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def entities(record):
    value = record["PII_set"]
    return ast.literal_eval(value) if isinstance(value, str) else value


def main(train_json, out_jsonl, out_manifest):
    raw = Path(train_json).read_bytes()
    if hashlib.md5(raw).hexdigest() != TRAIN_MD5:
        raise ValueError("official train.json checksum differs")
    records = json.loads(raw)
    evaluation = set()
    for name in ("kdpii_test_500", "kdpii_fresh_1992"):
        evaluation |= set(json.loads(Path(
            f"benchmarks/results/name-p0-ledger-hashes/{name}.json").read_text())[
                "id_to_text_sha256"].values())
    pool = [r for r in records if sha(r["sentence"]) not in evaluation]
    string_records = collections.defaultdict(set)
    for r in pool:
        for e in entities(r):
            string_records[e["form"]].add(r["sent_idx"])
    used_groups, chosen, shortfall = set(), [], {}
    for stratum, needed, quota, role in PLAN:
        labels_of = lambda r: {e["label"] for e in entities(r)}  # noqa: E731
        eligible = [r for r in pool if needed <= labels_of(r) and (
            role == "personal" or not labels_of(r) & FORBIDDEN_FOR_NEGATIVE)]
        eligible.sort(key=lambda r: sha(SALT + r["sent_idx"]))
        taken = 0
        for r in eligible:
            group = r["sent_idx"].split("_")[0]
            if group in used_groups:
                continue
            targets = [e for e in entities(r) if e["label"] in needed]
            if len(targets) != 1:  # one unambiguous target span per sentence
                continue
            used_groups.add(group)
            e = targets[0]
            chosen.append(dict(
                id=r["sent_idx"], conversation_prefix=group, stratum=stratum, text=r["sentence"],
                target=dict(start=e["begin"], end=e["end"], form=e["form"],
                            source_label=e["label"]),
                other_source_entities=[dict(label=x["label"], start=x["begin"], end=x["end"])
                                       for x in entities(r) if x is not e],
                source_role=role,
                target_string_in_other_records=len(string_records[e["form"]] - {r["sent_idx"]}),
                policy_label="unreviewed", review=dict(verdict=None, rationale=None, reviewers=[])))
            taken += 1
            if taken == quota:
                break
        shortfall[stratum] = quota - taken
    Path(out_jsonl).write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chosen))
    manifest = dict(
        scope="name_handle_review_pilot_v1", status="development pilot; unreviewed",
        source="KDPII v1 official train.json (Zenodo 10968609, CC-BY-4.0), md5 " + TRAIN_MD5,
        selection=f"SHA-256 rank of '{SALT}'+sent_idx within stratum; one record per ID prefix; "
                  "evaluation-identical texts excluded; sentences with one target span only",
        plan={s: q for s, _, q, _ in PLAN}, shortfall=shortfall, rows=len(chosen),
        per_stratum=dict(collections.Counter(c["stratum"] for c in chosen)),
        distinct_conversation_prefixes=len({c["conversation_prefix"] for c in chosen}),
        distinct_target_strings=len({c["target"]["form"] for c in chosen}),
        sha256_jsonl=hashlib.sha256(Path(out_jsonl).read_bytes()).hexdigest(),
        excluded_evaluation_identical=len(records) - len(pool),
        note="source role is derived from KDPII labels only; policy_label needs independent review")
    Path(out_manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:4])
