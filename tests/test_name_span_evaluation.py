"""Protect original coordinates, unseen-source sampling, and privacy coverage gates."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from evaluate_name_span_external import paired_f1_interval, parse_klue, select_cases
    from train_name_span_experiment import batch, gate
finally:
    sys.path[:] = previous_path


def test_klue_parser_keeps_spaces_and_adjacent_single_character_names():
    source = "## klue-ner-test\tannotation\n김\tB-PS\n \tO\n이\tB-PS\n나\tI-PS\n다\tB-PS\n"
    (case,) = parse_klue(source)
    assert case["text"] == "김 이나다"
    assert [(e["start"], e["end"]) for e in case["expected"]] == [(0, 1), (2, 4), (4, 5)]


def test_parser_rejects_orphan_inside_person_label():
    with pytest.raises(ValueError, match="Orphan"):
        parse_klue("## klue-ner-test\tannotation\n김\tI-PS\n")


def test_external_sample_does_not_depend_on_labels_or_document_order():
    cases = [dict(id=str(i), expected=[], text="a") for i in range(20)]
    chosen = [c["id"] for c in select_cases(cases, 6)]
    changed = [dict(c, expected=[dict(entity="KR_NAME", start=0, end=1)]) for c in cases[::-1]]
    assert [c["id"] for c in select_cases(changed, 6)] == chosen


def test_gate_protects_coverage_even_when_old_boundaries_were_wrong():
    cases = [dict(id="person", expected=[dict(entity="KR_NAME", start=1, end=3)])]
    result = gate(cases, [{(0, 4)}], [{(1, 2)}])
    assert result["lost_correct_spans"] == []
    assert result["newly_exposed_names"] == [("person", 1, 3)]
    assert not result["passed"]


def test_gate_forbids_new_false_span_despite_recovering_a_name():
    cases = [dict(id="person", expected=[dict(entity="KR_NAME", start=1, end=3)])]
    result = gate(cases, [set()], [{(1, 3), (4, 5)}])
    assert result["lost_correct_spans"] == result["newly_exposed_names"] == []
    assert result["introduced_false_spans"] == [("person", 4, 5)]
    assert not result["passed"]


def test_training_does_not_silently_drop_names_above_candidate_width():
    import torch

    case = dict(text="가나다", expected=[dict(entity="KR_NAME", start=0, end=3)])
    rows = [(torch.zeros(3, 8), None, torch.zeros(3, 2), None)]
    with pytest.raises(ValueError, match="exceeds"):
        batch(rows, [case], {}, 2)


def test_paired_bootstrap_is_zero_for_identical_predictions():
    cases = [dict(id=str(i), expected=[dict(entity="KR_NAME", start=0, end=1)]) for i in range(3)]
    predictions = [{(0, 1)}] * 3
    assert paired_f1_interval(cases, predictions, predictions, repetitions=40)[
        "f1_delta_95_interval"
    ] == [0.0, 0.0]
