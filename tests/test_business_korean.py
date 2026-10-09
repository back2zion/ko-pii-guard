"""Business prose negative gates; contextual-model accuracy is evaluated separately."""

import json
from pathlib import Path

import pytest

from ko_pii_guard import KoreanPIIGuard

DATA = Path(__file__).resolve().parents[1] / "benchmarks/data/business_korean.jsonl"
CASES = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()]
NEGATIVES = [case for case in CASES if case["category"] == "negative"]
POSITIVES = [case for case in CASES if case["category"] == "positive"]


@pytest.fixture(scope="module", params=["default", "name_address_rules"])
def guard(request):
    if request.param == "default":
        return KoreanPIIGuard()
    return KoreanPIIGuard(entities=["KR_NAME", "KR_ADDRESS"])


@pytest.mark.parametrize("case", NEGATIVES, ids=lambda row: row["id"])
def test_business_prose_has_no_names_or_other_pii_without_values(guard, case):
    text = case["text"]
    assert guard.analyze(text) == []
    assert guard.mask(text) == text
    assert not guard.contains_pii(text)


def test_name_address_recognition_is_opt_in():
    guard = KoreanPIIGuard()
    assert "KR_NAME" not in guard.entities
    assert "KR_ADDRESS" not in guard.entities
    text = "성명: 오지은, 주소: 서울특별시 강남구 가상누리로 9999"
    assert guard.analyze(text) == []
    assert guard.mask(text) == text


def test_positive_annotations_remain_separate_from_negative_gates():
    # These are valid gold annotations, not an assertion that either backend
    # recognizes all business fields or every name in free prose.
    assert POSITIVES
    assert {row["track"] for row in POSITIVES} == {"structured", "free_prose"}
    assert all(row["track"] == "business_negative" for row in NEGATIVES)
    assert len({row["id"] for row in CASES}) == len(CASES)
    for row in POSITIVES:
        assert row["expected"]
        cursor = 0
        for result in row["expected"]:
            assert result["entity"] == "KR_NAME"
            assert cursor <= result["start"] < result["end"] <= len(row["text"])
            cursor = result["end"]
