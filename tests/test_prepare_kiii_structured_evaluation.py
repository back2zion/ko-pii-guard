"""Typed-field extraction must preserve the pinned source's exact surfaces."""

import pytest

from scripts.prepare_kiii_structured_evaluation import convert


def test_conversion_uses_explicit_schema_and_keeps_hard_negatives_separate():
    row = {"doc_id": "synthetic-one", "text": "010-1234-5678 ABC-123",
           "spans": [{"category": "phone_no", "start": 0, "end": 13,
                      "surface": "010-1234-5678"},
                     {"category": "customer_id", "start": 14, "end": 21,
                      "surface": "ABC-123"}],
           "hard_negatives": [{"start": 14, "end": 21, "surface": "ABC-123"}]}
    assert convert(row) == {
        "id": "synthetic-one",
        "record": {"fields": [{"phone_no": "010-1234-5678"},
                              {"reference": "ABC-123"}]},
        "targets": [{"path": ["fields", 0, "phone_no"], "type": "PHONE_NUMBER"}],
        "negative_paths": [["fields", 1, "reference"]],
        "skipped_pre_masked_targets": 0,
    }
    assert convert(row, include_hard_negatives=False)["negative_paths"] == []


def test_conversion_rejects_misaligned_source_surface():
    row = {"doc_id": "bad", "text": "010-1234-5678",
           "spans": [{"category": "phone_no", "start": 1, "end": 13,
                      "surface": "010-1234-5678"}], "hard_negatives": []}
    with pytest.raises(ValueError, match="invalid supported span"):
        convert(row)


def test_name_address_fields_are_explicitly_opted_in():
    text = "홍길동 서울특별시 강남구 테헤란로 123"
    row = {"doc_id": "name-address", "text": text,
           "spans": [{"category": "person_name", "start": 0, "end": 3,
                      "surface": "홍길동"},
                     {"category": "address", "start": 4, "end": len(text),
                      "surface": text[4:]}],
           "hard_negatives": []}
    assert convert(row)["targets"] == []
    assert convert(row, include_name_address=True)["targets"] == [
        {"path": ["fields", 0, "person_name"], "type": "KR_NAME"},
        {"path": ["fields", 1, "address"], "type": "KR_ADDRESS"},
    ]


@pytest.mark.parametrize("surface", ["***", "<PHONE_NUMBER>", "<KR_NAME>"])
def test_fully_pre_masked_source_span_is_not_scored_as_a_raw_value(surface):
    row = {"doc_id": "pre-masked", "text": surface,
           "spans": [{"category": "phone_no", "start": 0, "end": len(surface),
                      "surface": surface}], "hard_negatives": []}
    case = convert(row)
    assert case["targets"] == []
    assert case["skipped_pre_masked_targets"] == 1
