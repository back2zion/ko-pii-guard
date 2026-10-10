"""Identical annotation instances are redundant gold, not contradictory spans."""

import copy
import sys
from pathlib import Path

import pytest

previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import kdpii_lossless_annotations as lossless
    from fetch_kdpii_evaluation import convert_record
finally:
    sys.path[:] = previous_path


def record(label="CV_POSITION"):
    return dict(sent_idx="fixture", sentence="팀장 홍길동", sent_seq=list("팀장 홍길동"),
                labelling_seq=["B-" + label, "I-" + label, "O", "B-PS_NAME", "I-PS_NAME",
                               "I-PS_NAME"],
                PII_set=[dict(id=0, begin=0, end=2, label=label, form="팀장"),
                         dict(id=1, begin=3, end=6, label="PS_NAME", form="홍길동")])


def test_ordinary_records_preserve_every_existing_converter_field():
    source = record()
    old, new = convert_record(source), lossless.convert_record_lossless(source)
    assert {key: new[key] for key in old} == old
    assert new["original_PII_set"] == source["PII_set"]
    assert new["duplicate_annotations"] == []


def test_nonperson_duplicate_instances_preserved_and_bio_validated_without_new_gold():
    source = record()
    source["PII_set"].append(dict(source["PII_set"][0], id=2))
    before = copy.deepcopy(source)
    with pytest.raises(ValueError, match="Overlapping"):
        convert_record(source)
    result = lossless.convert_record_lossless(source)
    assert source == before
    assert result["original_PII_set"] == source["PII_set"]
    assert len(result["source_annotations"]) == 3
    assert result["expected"] == [dict(entity="KR_NAME", start=3, end=6)]
    assert result["duplicate_annotations"] == [dict(label="CV_POSITION", start=0, end=2,
                                                    indices=[0, 2])]
    lossless.validate_cases_lossless([result], ["fixture"])


@pytest.mark.parametrize("label,primary_count,expanded_count", [
    ("PS_NAME", 2, 2), ("PS_NICKNAME", 1, 2),
])
def test_repeated_name_or_nickname_is_one_gold_event(label, primary_count, expanded_count):
    source = record(label)
    source["PII_set"].append(dict(source["PII_set"][0], id=2))
    result = lossless.convert_record_lossless(source)
    assert len(result["source_annotations"]) == 3
    assert len(result["expected"]) == primary_count
    assert len(result["alternate_expected"]) == expanded_count
    assert len({(row["start"], row["end"]) for row in result["expected"]}) == primary_count
    lossless.validate_cases_lossless([result], ["fixture"])


@pytest.mark.parametrize("extra", [
    dict(id=2, begin=0, end=2, label="PS_NAME", form="팀장"),
    dict(id=2, begin=1, end=2, label="CV_POSITION", form="장"),
    dict(id=2, begin=1, end=4, label="CV_POSITION", form="장 홍"),
    dict(id=2, begin=0, end=2, label="CV_POSITION", form="틀림"),
])
def test_different_labels_nested_partial_and_wrong_surface_still_fail(extra):
    source = record()
    source["PII_set"].append(extra)
    with pytest.raises(ValueError):
        lossless.convert_record_lossless(source)


def test_identical_duplicates_cannot_bypass_original_full_bio_check():
    source = record()
    source["PII_set"].append(dict(source["PII_set"][0], id=2))
    source["labelling_seq"][0] = "B-PS_NAME"
    with pytest.raises(ValueError, match="BIO"):
        lossless.convert_record_lossless(source)


def test_validation_rejects_duplicated_gold_and_missing_original_annotation_instances():
    source = record("PS_NAME")
    source["PII_set"].append(dict(source["PII_set"][0], id=2))
    valid = lossless.convert_record_lossless(source)
    altered = copy.deepcopy(valid)
    altered["expected"].append(altered["expected"][0])
    with pytest.raises(ValueError, match="gold"):
        lossless.validate_cases_lossless([altered], ["fixture"])
    altered = copy.deepcopy(valid)
    altered["source_annotations"].pop()
    with pytest.raises(ValueError, match="preserv"):
        lossless.validate_cases_lossless([altered], ["fixture"])


def test_reader_preserves_fixed_order_and_never_converts_unselected_records(tmp_path):
    import json

    source = record()
    path = tmp_path / "test.json"
    path.write_text(json.dumps([dict(sent_idx="unselected", malformed="sealed"), source]))
    cases, duplicates = lossless.read_cases_lossless(path, ["fixture"])
    assert [case["id"] for case in cases] == ["fixture"]
    assert duplicates == []
    with pytest.raises(ValueError, match="selected"):
        lossless.read_cases_lossless(path, ["absent"])
