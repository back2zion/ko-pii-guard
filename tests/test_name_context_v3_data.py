"""V3 split isolation is checked before model inference."""

import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v3_is_reproducible_and_new_splits_are_isolated_from_all_inspected_data():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    try:
        generator = runpy.run_path(str(ROOT / "benchmarks/name_context_v3_cases.py"))
    finally:
        sys.path.pop(0)
    cases = [
        json.loads(line)
        for line in (ROOT / "benchmarks/data/name_context_v3.jsonl").read_text().splitlines()
    ]
    assert generator["build_cases"]() == cases
    assert len({c["id"] for c in cases}) == len(cases)
    splits = {s: [c for c in cases if c["split"] == s] for s in (
        "train", "validation", "evaluation"
    )}
    assert {s: len(rows) for s, rows in splits.items()} == {
        "train": 6520, "validation": 84, "evaluation": 280
    }
    for left, right in (("train", "validation"), ("train", "evaluation"),
                        ("validation", "evaluation")):
        for key in ("text", "context_id", "surface"):
            a = {c[key] for c in splits[left] if c[key] is not None}
            b = {c[key] for c in splits[right] if c[key] is not None}
            assert a.isdisjoint(b)
        def names(rows):
            return {c["text"][e["start"]:e["end"]] for c in rows for e in c["expected"]}
        assert names(splits[left]).isdisjoint(names(splits[right]))
    for case in cases:
        previous_end = 0
        for e in case["expected"]:
            assert e["entity"] == "KR_NAME"
            assert previous_end <= e["start"] < e["end"] <= len(case["text"])
            previous_end = e["end"]
    old = [json.loads(line) for line in (
        ROOT / "benchmarks/data/name_context_v2.jsonl"
    ).read_text().splitlines()]
    copied = {c["id"]: c for c in splits["train"]}
    for case in old:
        new = copied["v3-development-" + case["id"]]
        assert (new["text"], new["expected"]) == (case["text"], case["expected"])
    regressions = [json.loads(line) for line in (
        ROOT / "benchmarks/data/name_context_v2_regressions.jsonl"
    ).read_text().splitlines()]
    assert regressions == [case for case in old if case["split"] == "evaluation"]
