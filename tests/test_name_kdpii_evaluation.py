"""Keep KDPII name/nickname policies explicit without repeating inference."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from evaluate_name_kdpii import evaluate_policies, validate_cases
finally:
    sys.path[:] = previous_path


def cases():
    return [dict(
        id="sentence-1", text="김 배", expected=[dict(entity="KR_NAME", start=0, end=1)],
        alternate_expected=[dict(entity="KR_NAME", start=0, end=1),
                            dict(entity="KR_NAME", start=2, end=3)],
        source_annotations=[dict(label="PS_NAME", start=0, end=1),
                            dict(label="PS_NICKNAME", start=2, end=3)],
    )]


def test_broad_policy_uses_identical_predictions_and_masks_without_another_call():
    class Guard:
        analyze_calls = 0
        mask_calls = 0

        def analyze(self, text):
            self.analyze_calls += 1
            return [SimpleNamespace(entity="KR_NAME", start=0, end=1),
                    SimpleNamespace(entity="KR_NAME", start=2, end=3)]

        def mask(self, text, *, style):
            self.mask_calls += 1
            assert style == "stars"
            return "* *"

    guard = Guard()
    scores, predicted, masks = evaluate_policies(cases(), guard)
    assert guard.analyze_calls == guard.mask_calls == 1
    assert predicted == [{(0, 1), (2, 3)}] and masks == ["* *"]
    assert scores["PS_NAME"]["tp"] == scores["PS_NAME"]["fp"] == 1
    assert scores["PS_NAME+PS_NICKNAME"]["tp"] == 2
    assert scores["PS_NAME+PS_NICKNAME"]["fp"] == 0
    assert scores["PS_NAME"]["fully_covered_names"] == 1
    assert scores["PS_NAME+PS_NICKNAME"]["fully_covered_names"] == 2
    assert list(scores["PS_NAME"]["by_source"]) == ["kdpii_v1_test"]


def test_policy_mapping_is_verified_against_preserved_source_labels():
    rows = cases()
    validate_cases(rows, ["sentence-1"])
    rows[0]["alternate_expected"] = rows[0]["expected"]
    with pytest.raises(ValueError, match="PS_NAME\\+PS_NICKNAME"):
        validate_cases(rows, ["sentence-1"])


def test_selected_records_cannot_be_dropped_after_annotation_inspection():
    with pytest.raises(ValueError, match="selected IDs"):
        validate_cases(cases(), ["sentence-1", "sentence-2"])
