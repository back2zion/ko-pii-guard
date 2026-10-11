"""Validate and merge two blind agent reviews of the nickname/handle pilot.

Inputs (outside the repo during review): reviewer dirs with part1..4.jsonl, the review items,
the review key and the pilot file. Outputs: merged reviews with original verdicts preserved, an
adjudication queue, a stratified 10% audit sample of agreed items, and agreement statistics.
Source labels are joined only here, after both reviews are fixed, and only for aggregate reporting.
"""

from __future__ import annotations

import collections
import hashlib
import json
import random
import sys
from pathlib import Path

VERDICTS = {"personal", "non_personal", "insufficient"}
AUDIT_SEED, AUDIT_FRACTION = 20261031, 0.10


def load_reviewer(directory):
    rows = {}
    for part in sorted(Path(directory).glob("part*.jsonl")):
        for line in part.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"duplicate {row['id']} in {directory}")
            rows[row["id"]] = row
    return rows


def validate(rows, items):
    problems = []
    if set(rows) != set(items):
        problems.append(f"id mismatch: missing {sorted(set(items) - set(rows))[:5]} "
                        f"extra {sorted(set(rows) - set(items))[:5]}")
    for rid, row in rows.items():
        if row.get("verdict") not in VERDICTS:
            problems.append(f"{rid}: bad verdict {row.get('verdict')!r}")
        if rid in items and row.get("evidence") and row["evidence"] not in items[rid]["text"]:
            problems.append(f"{rid}: evidence is not a verbatim substring")
        if row.get("policy_pending") and not row.get("pending_question"):
            problems.append(f"{rid}: policy_pending without question")
    return problems


def kappa(pairs):
    labels = sorted({x for p in pairs for x in p})
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    expected = sum((sum(a == k for a, _ in pairs) / n) * (sum(b == k for _, b in pairs) / n)
                   for k in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def main(root, output_dir):
    root, output_dir = Path(root), Path(output_dir)
    items = {json.loads(line)["id"]: json.loads(line)
             for line in (root / "input/items.jsonl").read_text().splitlines()}
    a, b = load_reviewer(root / "reviewerA"), load_reviewer(root / "reviewerB")
    problems = {"A": validate(a, items), "B": validate(b, items)}
    key = {m["review_id"]: m["pilot_id"] for m in json.loads(
        Path("benchmarks/data/name_handle_pilot_v1.review-key.json").read_text())["mapping"]}
    pilot = {json.loads(line)["id"]: json.loads(line) for line in
             Path("benchmarks/data/name_handle_pilot_v1.jsonl").read_text().splitlines()}
    merged, queue, agreed = [], [], collections.defaultdict(list)
    for rid in sorted(items):
        ra, rb = a.get(rid), b.get(rid)
        if ra is None or rb is None:
            continue
        reasons = []
        if ra["verdict"] != rb["verdict"]:
            reasons.append("disagreement")
        if "insufficient" in (ra["verdict"], rb["verdict"]):
            reasons.append("insufficient")
        if ra.get("policy_pending") or rb.get("policy_pending"):
            reasons.append("policy_pending")
        if not (ra.get("span_ok", True) and rb.get("span_ok", True)):
            reasons.append("span_questioned")
        row = dict(review_id=rid, pilot_id=key[rid], reviewer_A=ra, reviewer_B=rb,
                   status="needs_adjudication" if reasons else "agent_consensus",
                   queue_reasons=reasons)
        merged.append(row)
        (queue if reasons else agreed[ra["verdict"]]).append(row)
    rng = random.Random(AUDIT_SEED)
    audit = []
    for _, rows in sorted(agreed.items()):
        rows = sorted(rows, key=lambda r: r["review_id"])
        take = max(1, round(len(rows) * AUDIT_FRACTION)) if rows else 0
        audit.extend(rng.sample(rows, take))
    # Agreed verdicts that contradict the (hidden) source role are always shown to the adjudicator.
    contradiction = []
    for rows in agreed.values():
        for row in rows:
            role = pilot[row["pilot_id"]]["source_role"]
            if role != row["reviewer_A"]["verdict"] and row not in audit:
                contradiction.append(row)
    audit.extend(contradiction)
    audit_ids = {r["review_id"] for r in audit}
    for row in merged:
        if row["review_id"] in audit_ids:
            row["status"] = "agent_consensus_audit_pending"
    pairs = [(a[r]["verdict"], b[r]["verdict"]) for r in items if r in a and r in b]
    source = collections.defaultdict(collections.Counter)
    for row in merged:
        p = pilot[row["pilot_id"]]
        both = row["reviewer_A"]["verdict"] if row["status"] != "needs_adjudication" else "queue"
        source[p["stratum"]][both] += 1
    stats = dict(
        items=len(items), validation_problems=problems,
        agreement=sum(x == y for x, y in pairs) / len(pairs),
        cohen_kappa=round(kappa(pairs), 4),
        verdict_counts=dict(A=collections.Counter(r["verdict"] for r in a.values()),
                            B=collections.Counter(r["verdict"] for r in b.values())),
        policy_pending=dict(A=sum(bool(r.get("policy_pending")) for r in a.values()),
                            B=sum(bool(r.get("policy_pending")) for r in b.values())),
        span_not_ok=dict(A=sum(r.get("span_ok") is False for r in a.values()),
                         B=sum(r.get("span_ok") is False for r in b.values())),
        adjudication_queue=len(queue),
        queue_reasons=dict(collections.Counter(x for r in queue for x in r["queue_reasons"])),
        agreed=sum(len(v) for v in agreed.values()), audit_sample=len(audit),
        audit_extra_source_contradictions=len(contradiction),
        audit_sample_by_verdict=dict(
            collections.Counter(r["reviewer_A"]["verdict"] for r in audit)),
        result_by_source_stratum={k: dict(v) for k, v in sorted(source.items())},
        note="stratum counts are aggregate only; reviewers never saw source labels",
        audit_seed=AUDIT_SEED, audit_fraction=AUDIT_FRACTION)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "name_handle_pilot_v1.reviews.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged))
    (output_dir / "name_handle_pilot_v1.adjudication_queue.jsonl").write_text(
        "".join(json.dumps(dict(review_id=r["review_id"], text=items[r["review_id"]]["text"],
                                span=items[r["review_id"]]["span"], reasons=r["queue_reasons"],
                                A=r["reviewer_A"], B=r["reviewer_B"]), ensure_ascii=False) + "\n"
                for r in queue))
    (output_dir / "name_handle_pilot_v1.audit_sample.jsonl").write_text(
        "".join(json.dumps(dict(review_id=r["review_id"], text=items[r["review_id"]]["text"],
                                span=items[r["review_id"]]["span"], A=r["reviewer_A"],
                                B=r["reviewer_B"]), ensure_ascii=False) + "\n" for r in audit))
    stats["input_sha256"] = {n: hashlib.sha256((root / n).read_bytes()).hexdigest()
                             for n in ("input/items.jsonl", "input/guideline.md")}
    (output_dir / "name_handle_pilot_v1.review-summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:3])
