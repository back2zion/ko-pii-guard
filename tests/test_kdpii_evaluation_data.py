"""Original sentence offsets and label-independent KDPII sampling."""

import sys
from pathlib import Path

import pytest

previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from fetch_kdpii_evaluation import convert_record, select_ids, verify_metadata
finally:
    sys.path[:] = previous_path


def test_name_nickname_offsets_preserve_spaces_and_repeated_forms():
    record = dict(
        sent_idx="a",
        sentence="봄 봄이와 별이",
        sent_seq=list("봄 봄이와 별이"),
        PII_set=[
            dict(form="봄", label="PS_NAME", begin=2, end=3),
            dict(form="별이", label="PS_NICKNAME", begin=6, end=8),
        ],
        labelling_seq=["O", "O", "B-PS_NAME", "O", "O", "O", "B-PS_NICKNAME", "I-PS_NICKNAME"],
    )
    case = convert_record(record)
    assert case["text"] == record["sentence"]
    assert case["expected"] == [dict(entity="KR_NAME", start=2, end=3)]
    assert case["alternate_expected"] == [
        dict(entity="KR_NAME", start=2, end=3),
        dict(entity="KR_NAME", start=6, end=8),
    ]
    assert case["source_annotations"][1]["label"] == "PS_NICKNAME"


@pytest.mark.parametrize(
    "change",
    [
        {"sentence": "김 가"},
        {"PII_set": [dict(form="김", label="PS_NAME", begin=1, end=2)]},
        {"labelling_seq": ["O"]},
    ],
)
def test_offset_surface_and_character_labels_must_agree(change):
    record = dict(
        sent_idx="a",
        sentence="김",
        sent_seq=["김"],
        PII_set=[dict(form="김", label="PS_NAME", begin=0, end=1)],
        labelling_seq=["B-PS_NAME"],
    )
    with pytest.raises(ValueError):
        convert_record(dict(record, **change))


def test_handles_are_outside_person_name_mapping():
    record = dict(
        sent_idx="a",
        sentence="별",
        sent_seq=["별"],
        PII_set=[dict(form="별", label="PS_ID", begin=0, end=1)],
        labelling_seq=["B-PS_ID"],
    )
    assert convert_record(record)["expected"] == []


def test_sampling_depends_only_on_id_and_rejects_duplicate_ids():
    ids = [str(i) for i in range(20)]
    assert select_ids(ids, 5) == select_ids(ids[::-1], 5)
    with pytest.raises(ValueError, match="ID"):
        select_ids(["a", "a"], 1)


def test_wrong_license_metadata_is_rejected():
    with pytest.raises(ValueError, match="license"):
        verify_metadata(dict(id=10968609, metadata=dict(license=dict(id="unknown")), files=[]))
