"""Curated regression checks; these examples are not held-out evaluation data."""

import json
from pathlib import Path

import pytest

from ko_pii_guard import KoreanPIIGuard

DATA = Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "robustness.jsonl"
CASES = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def guard():
    return KoreanPIIGuard()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_curated_spans_and_masking(guard, case):
    text = case["text"]
    findings = guard.analyze(text)
    expected = [(e["entity"], e["start"], e["end"]) for e in case["expected"]]
    assert [(f.entity, f.start, f.end) for f in findings] == expected
    assert all(f.text == text[f.start:f.end] for f in findings)
    assert guard.contains_pii(text) is bool(expected)

    stars = list(text)
    for _, start, end in expected:
        stars[start:end] = "*" * (end - start)
    assert guard.mask(text, style="stars") == "".join(stars)

    tagged = text
    for entity, start, end in reversed(expected):
        tagged = tagged[:start] + f"<{entity}>" + tagged[end:]
    assert guard.mask(text) == tagged


def test_corpus_annotations_are_well_formed():
    assert len({case["id"] for case in CASES}) == len(CASES)
    for case in CASES:
        cursor = 0
        for span in case["expected"]:
            assert cursor <= span["start"] < span["end"] <= len(case["text"])
            cursor = span["end"]
