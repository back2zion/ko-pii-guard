"""The v11 evaluation stays separate from every inspected case and gold name."""

import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v11_is_frozen_reproducible_and_preserves_all_inspected_annotations():
    previous_path = sys.path[:]
    sys.path.insert(0, str(ROOT / "benchmarks"))
    try:
        api = runpy.run_path(str(ROOT / "benchmarks/name_context_v11_cases.py"))
    finally:
        sys.path[:] = previous_path
    cases = [json.loads(line) for line in (
        ROOT / "benchmarks/data/name_context_v11.jsonl"
    ).read_text().splitlines()]
    assert api["build_cases"]() == cases
    assert len({c["id"] for c in cases}) == len(cases)
    splits = {s: [c for c in cases if c["split"] == s]
              for s in ("train", "validation", "evaluation")}
    assert {s: len(rows) for s, rows in splits.items()} == {
        "train": 10056, "validation": 96, "evaluation": 192
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
    originals = [json.loads(line) for line in (
        ROOT / "benchmarks/data/name_context_v10.jsonl"
    ).read_text().splitlines()]
    copied = {c["id"]: c for c in splits["train"]}
    for old in originals:
        new = copied["v11-development-" + old["id"]]
        assert (old["text"], old["expected"]) == (new["text"], new["expected"])
    for case in cases:
        last = 0
        for span in case["expected"]:
            assert span["entity"] == "KR_NAME"
            assert last <= span["start"] < span["end"] <= len(case["text"])
            last = span["end"]
