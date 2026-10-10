"""Keep held-out text sealed and check actual masking regressions."""

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from evaluate_name_generalization import (
        actual_mask_gate,
        collect_ids,
        evaluate_guard,
        load_training_text_hashes,
        make_split,
        paired_intervals,
        read_selected_cases,
        training_overlap_ids,
        validate_devices,
        validate_selection,
        validate_split,
    )
finally:
    sys.path[:] = previous_path


def test_split_only_reads_headers_and_ignores_unselected_annotations(tmp_path):
    source = tmp_path / "source.tsv"
    source.write_text("## klue-ner-1-source\tunknown\nINVALID SECRET LABEL\n")
    assert collect_ids(source) == ["klue-ner-1-source"]


def test_split_is_disjoint_and_independent_of_input_order():
    ids = [f"klue-ner-{i}-source" for i in range(12)]
    previous = ids[:3]
    first = make_split(ids, previous, 4)
    second = make_split(ids[::-1], previous[::-1], 4)
    assert first == second
    assert len(first["heldout_ids"]) == 4
    assert not set(first["heldout_ids"]) & set(previous)
    assert set(first["development_ids"]) == set(previous)


def test_split_rejects_duplicate_or_unknown_ids():
    with pytest.raises(ValueError, match="duplicate"):
        make_split(["a", "a"], [], 1)
    with pytest.raises(ValueError, match="unknown"):
        make_split(["a", "b"], ["z"], 1)


def test_changed_but_disjoint_holdout_ids_are_rejected():
    ids = [f"sample-{i}" for i in range(20)]
    split = make_split(ids, ids[:3], 4)
    validate_split(split, ids, ids[:3])
    replacement = next(i for i in ids if i not in split["heldout_ids"] + ids[:3])
    split["heldout_ids"][0] = replacement
    with pytest.raises(ValueError, match="predeclared"):
        validate_split(split, ids, ids[:3])


def test_selected_parser_does_not_parse_other_records(tmp_path):
    source = tmp_path / "source.tsv"
    source.write_text(
        "## klue-ner-sealed-source\tunknown\nINVALID LABEL\n"
        "## klue-ner-selected-source\tannotation\n김\tB-PS\n \tO\n"
    )
    (case,) = read_selected_cases(source, ["klue-ner-selected-source"])
    assert case["text"] == "김 "
    assert case["expected"] == [{"entity": "KR_NAME", "start": 0, "end": 1}]


def test_final_selection_requires_frozen_unchanged_inputs(tmp_path):
    model = tmp_path / "model.json"
    model.write_text("selected")
    selection = {
        "finalized": True,
        "selection_basis": "development only",
        "candidate": {"mode": "e5"},
        "frozen_sha256": {str(model): hashlib.sha256(model.read_bytes()).hexdigest()},
    }
    validate_selection(selection, heldout=True)
    selection["finalized"] = False
    with pytest.raises(ValueError, match="finalized"):
        validate_selection(selection, heldout=True)
    selection["finalized"] = True
    model.write_text("changed after selection")
    with pytest.raises(ValueError, match="changed"):
        validate_selection(selection, heldout=True)


def test_learned_holdout_requires_frozen_training_text_hashes(tmp_path):
    selection = {"candidate": {}, "frozen_sha256": {}}
    with pytest.raises(ValueError, match="training_text_hashes_json"):
        load_training_text_hashes(selection, heldout=True)
    source = tmp_path / "hashes.json"
    source.write_text('{"algorithm":"sha256_utf8_exact_text","text_sha256":[]}')
    selection["training_text_hashes_json"] = str(source)
    with pytest.raises(ValueError, match="frozen"):
        load_training_text_hashes(selection, heldout=True)
    selection["frozen_sha256"][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    assert load_training_text_hashes(selection, heldout=True) == set()


def test_training_overlap_reports_exact_text_ids_without_resampling():
    cases = [dict(id="overlap", text="김이 말"), dict(id="distinct", text="김 이 말")]
    hashes = {hashlib.sha256("김이 말".encode()).hexdigest()}
    assert training_overlap_ids(cases, hashes) == ["overlap"]
    assert [case["id"] for case in cases] == ["overlap", "distinct"]


def test_device_mismatch_rejected_and_backbone_device_resolved():
    baseline = SimpleNamespace(device="cpu")
    candidate = SimpleNamespace(backbone=SimpleNamespace(device="cuda:0"))
    with pytest.raises(ValueError, match="devices differ"):
        validate_devices(baseline, candidate)
    candidate.backbone.device = "cpu"
    assert validate_devices(baseline, candidate) == {"baseline": "cpu", "candidate": "cpu"}


def test_actual_mask_metrics_use_mask_api_and_count_masked_non_person_chars():
    cases = [dict(id="klue-ner-1-source", text="김이 말", expected=[
        dict(entity="KR_NAME", start=0, end=2),
    ])]

    class Guard:
        def analyze(self, text):
            return [SimpleNamespace(entity="KR_NAME", start=0, end=2)]

        def mask(self, text, *, style):
            assert style == "stars"
            return "*이 *"

    score, predictions, masks = evaluate_guard(cases, Guard())
    assert predictions == [{(0, 2)}]
    assert masks == ["*이 *"]
    assert score["tp"] == 1
    assert score["fully_covered_names"] == 0
    assert score["unnecessary_masked_characters"] == 1
    assert score["by_source"]["source"]["fully_covered_names"] == 0


def test_actual_mask_gate_protects_coverage_despite_identical_analyze_output():
    cases = [dict(id="a", text="김이", expected=[dict(entity="KR_NAME", start=0, end=2)])]
    predictions = [{(0, 2)}]
    result = actual_mask_gate(cases, predictions, predictions, ["**"], ["*이"])
    assert not result["passed"]
    assert result["lost_correct_spans"] == []
    assert result["actual_stars_newly_exposed_names"] == [("a", 0, 2)]


def test_bootstrap_identical_predictions_and_masks_has_zero_differences():
    cases = [dict(id="a", text="김", expected=[dict(entity="KR_NAME", start=0, end=1)])]
    result = paired_intervals(cases, [{(0, 1)}], [{(0, 1)}], ["*"], ["*"], repetitions=20)
    assert result["delta_95_intervals"] == {
        "f1": [0.0, 0.0], "precision": [0.0, 0.0], "recall": [0.0, 0.0],
        "full_name_coverage": [0.0, 0.0],
    }
