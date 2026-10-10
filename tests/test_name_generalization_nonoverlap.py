"""Contamination exclusion cannot change the model, inspect scores, or resample."""

import copy
import hashlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from evaluate_name_generalization_nonoverlap import (
        exclude_exact_training_overlap,
        validate_blocked_report,
    )
finally:
    sys.path[:] = previous_path


def blocked_fixture():
    selection = dict(candidate=dict(checkpoint="fixed-model", threshold=0.01))
    return selection, dict(
        scope="heldout", source_changed_during_run=False,
        evaluation_blocked="heldout_training_or_validation_text_overlap",
        selected_ids=["a", "b", "c"], candidate_factory="module:factory",
        selection=copy.deepcopy(selection),
        exact_training_or_validation_text_overlap_ids=["b"],
    )


def test_only_exact_overlap_is_removed_without_replacement_or_reordering():
    rows = [dict(id="a", text="김 이", expected=[]), dict(id="b", text="김이", expected=[]),
            dict(id="c", text="홍길동", expected=[])]
    hashes = {hashlib.sha256("김이".encode()).hexdigest()}
    retained, excluded = exclude_exact_training_overlap(rows, hashes)
    assert [row["id"] for row in retained] == ["a", "c"]
    assert excluded == ["b"]
    assert rows[1]["id"] == "b"
    rows[0]["expected"] = [{"entity": "KR_NAME", "start": 0, "end": 1}]
    assert exclude_exact_training_overlap(rows, hashes) == (retained, excluded)


def test_amendment_requires_original_pre_inference_block_for_identical_candidate():
    selection, blocked = blocked_fixture()
    validate_blocked_report(blocked, selection, "module:factory", ["a", "b", "c"])
    selection["candidate"]["threshold"] = 0.1
    with pytest.raises(ValueError, match="same candidate"):
        validate_blocked_report(blocked, selection, "module:factory", ["a", "b", "c"])


def test_original_selected_ids_cannot_be_resampled():
    selection, blocked = blocked_fixture()
    with pytest.raises(ValueError, match="original selected IDs"):
        validate_blocked_report(blocked, selection, "module:factory", ["a", "b", "new"])


def test_already_inferred_holdout_cannot_be_amended_as_pre_inference_exclusion():
    selection, blocked = blocked_fixture()
    blocked["candidate"] = {"metrics": {"tp": 1}}
    with pytest.raises(ValueError, match="before inference"):
        validate_blocked_report(blocked, selection, "module:factory", ["a", "b", "c"])


def test_missing_hashes_or_removing_entire_sample_is_rejected():
    rows = [dict(id="a", text="same")]
    with pytest.raises(ValueError, match="training hashes"):
        exclude_exact_training_overlap(rows, None)
    with pytest.raises(ValueError, match="empty"):
        exclude_exact_training_overlap(rows, {hashlib.sha256(b"same").hexdigest()})
