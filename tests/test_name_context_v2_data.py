"""V2 corpus provenance and selection-data isolation, without loading a model."""

import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
try:
    API = runpy.run_path(str(ROOT / "benchmarks/name_context_v2_cases.py"))
finally:
    sys.path.pop(0)
CASES = [
    json.loads(line)
    for line in (ROOT / "benchmarks/data/name_context_v2.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
]


def names(cases):
    return {c["text"][e["start"] : e["end"]] for c in cases for e in c["expected"]}


def test_v2_generator_is_reproducible_and_preserves_exact_multi_person_spans():
    assert API["build_cases"]() == CASES
    assert len({c["id"] for c in CASES}) == len(CASES)
    for case in CASES:
        previous_end = 0
        for span in case["expected"]:
            assert span["entity"] == "KR_NAME"
            assert type(span["start"]) is int and type(span["end"]) is int
            assert previous_end <= span["start"] < span["end"] <= len(case["text"])
            previous_end = span["end"]
        assert sorted(names([case])) == case["surfaces"]


def test_v2_all_gold_names_and_contexts_are_disjoint_between_splits():
    splits = {
        s: [c for c in CASES if c["split"] == s] for s in ("train", "validation", "evaluation")
    }
    assert all(splits.values())
    assert sum(map(len, splits.values())) == len(CASES)
    for left, right in [
        ("train", "validation"),
        ("train", "evaluation"),
        ("validation", "evaluation"),
    ]:
        assert names(splits[left]).isdisjoint(names(splits[right]))
        for key in ("text", "context_id"):
            assert {c[key] for c in splits[left]}.isdisjoint(c[key] for c in splits[right])
    for rows in splits.values():
        assert any(len(c["expected"]) >= 2 for c in rows)
        assert any(c["expected"] and ".pdf" in c["text"] for c in rows)
        assert any(e["end"] - e["start"] == 1 for c in rows for e in c["expected"])
        assert any(not c["expected"] and ".pdf" in c["text"] for c in rows)


def test_every_inspected_v1_case_is_now_training_and_originals_are_untouched():
    old = [
        json.loads(line)
        for line in (ROOT / "benchmarks/data/name_context_training.jsonl").read_text().splitlines()
    ]
    training = {c["id"]: c for c in CASES if c["split"] == "train"}
    for case in old:
        copied = training["v2-development-name_context_training.jsonl-" + case["id"]]
        assert copied["text"] == case["text"]
        assert copied["expected"] == case["expected"]
    assert any(c["split"] == "evaluation" for c in old)
