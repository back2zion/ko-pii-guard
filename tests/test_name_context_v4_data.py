"""Reproducible v4 corpus and evaluation isolation."""

import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v4_data_is_reproducible_preserves_v3_and_has_disjoint_new_splits():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    try:
        api = runpy.run_path(str(ROOT / "benchmarks/name_context_v4_cases.py"))
    finally:
        sys.path.pop(0)
    cases = [json.loads(line) for line in (
        ROOT / "benchmarks/data/name_context_v4.jsonl"
    ).read_text().splitlines()]
    assert cases == api["build_cases"]()
    assert len({c["id"] for c in cases}) == len(cases)
    splits = {s: [c for c in cases if c["split"] == s]
              for s in ("train", "validation", "evaluation")}
    assert {s: len(rows) for s, rows in splits.items()} == {
        "train": 8038, "validation": 68, "evaluation": 222
    }
    for left, right in (("train", "validation"), ("train", "evaluation"),
                        ("validation", "evaluation")):
        for key in ("text", "surface", "context_id"):
            assert {c[key] for c in splits[left] if c[key] is not None}.isdisjoint(
                c[key] for c in splits[right] if c[key] is not None
            )
        def names(rows):
            return {c["text"][e["start"]:e["end"]] for c in rows for e in c["expected"]}
        assert names(splits[left]).isdisjoint(names(splits[right]))
    old = [json.loads(line) for line in (
        ROOT / "benchmarks/data/name_context_v3.jsonl"
    ).read_text().splitlines()]
    copied = {c["id"]: c for c in splits["train"]}
    for case in old:
        new = copied["v4-development-" + case["id"]]
        assert (case["text"], case["expected"]) == (new["text"], new["expected"])
    for case in cases:
        last = 0
        for e in case["expected"]:
            assert e["entity"] == "KR_NAME"
            assert last <= e["start"] < e["end"] <= len(case["text"])
            last = e["end"]
    for split in splits.values():
        assert any('"' in c["text"] and c["expected"] for c in split)
        assert any(".pdf" in c["text"] and not c["expected"] for c in split)
