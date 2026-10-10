"""Replay preserves consumed input provenance and real public masking semantics."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
RecognizerResult = pytest.importorskip("presidio_analyzer").RecognizerResult
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from evaluate_name_generalization import evaluate_guard
    from evaluate_name_span_evidence_development import (
        ReplayNER,
        assert_baseline_replay,
        compact,
        digest,
        guard_for,
        validate_cache_cases,
        validate_cache_provenance,
        validate_pinned_file,
    )
finally:
    sys.path[:] = previous_path

def case(identifier="one", text="홍길동 왔다"):
    return {"id": identifier, "text": text, "expected": [
        {"entity": "KR_NAME", "start": 0, "end": 3}
    ]}


def test_identical_duplicate_text_keeps_both_case_ids_and_actual_masks():
    cases = [case("first"), case("second")]
    rows = [[RecognizerResult("KR_NAME", 0, 3, 1.)] for _ in cases]
    score, predictions, masks = evaluate_guard(cases, guard_for(cases, rows))
    assert score["tp"] == score["fully_covered_names"] == 2
    assert score["sentences"] == 2
    assert predictions == [{(0, 3)}, {(0, 3)}]
    assert masks == ["*** 왔다", "*** 왔다"]


def test_duplicate_text_with_conflicting_outputs_and_row_count_mismatch_fail():
    cases = [case("first"), case("second")]
    with pytest.raises(ValueError, match="Conflicting"):
        ReplayNER(cases, [[RecognizerResult("KR_NAME", 0, 3, 1.)], []])
    with pytest.raises(ValueError):
        ReplayNER(cases, [[]])


def test_identity_normalization_and_complete_character_logits_are_required():
    cases = [case()]
    cache = {"selected_ids": ["one"], "logits": [torch.zeros(len(cases[0]["text"]), 5)]}
    validate_cache_cases(cache, cases)
    with pytest.raises(ValueError, match="normalization"):
        validate_cache_cases({"selected_ids": ["one"], "logits": [torch.zeros(2, 5)]},
                             [case(text="Ａ씨")])
    with pytest.raises(ValueError, match="logits"):
        validate_cache_cases(dict(cache, logits=[torch.zeros(2, 5)]), cases)
    with pytest.raises(ValueError, match="logits"):
        validate_cache_cases(dict(cache, logits=[]), cases)
    with pytest.raises(ValueError, match="IDs"):
        validate_cache_cases(dict(cache, selected_ids=["other"]), cases)


def test_pinned_file_requires_exact_content_and_recorded_path(tmp_path):
    path = tmp_path / "consumed.json"
    path.write_text("original")
    report = {"frozen_sha256": {str(path.resolve()): digest(path)}}
    validate_pinned_file(report, path)
    path.write_text("changed")
    with pytest.raises(ValueError):
        validate_pinned_file(report, path)
    with pytest.raises(ValueError):
        validate_pinned_file({"frozen_sha256": {}}, path)


@pytest.mark.parametrize("scope", ["development_posterior_coverage_selection",
                                  "consumed_development_cache"])
def test_both_completed_cache_provenance_formats_pin_both_outputs(tmp_path, scope):
    cache, spans = tmp_path / "cache.pt", tmp_path / "spans.jsonl"
    for path in (cache, spans):
        path.write_text("fixed cache fixture")
    report = {"scope": scope, "source_changed_during_run": False, "source_sha256": {
        str(path.resolve()): digest(path) for path in (cache, spans)
    }}
    validate_cache_provenance(report, cache, spans)
    with pytest.raises(ValueError):
        validate_cache_provenance(dict(report, source_changed_during_run=True), cache, spans)
    with pytest.raises(ValueError):
        validate_cache_provenance(dict(report, scope="heldout"), cache, spans)
    spans.write_text("mutated spans")
    with pytest.raises(ValueError):
        validate_cache_provenance(report, cache, spans)


def test_baseline_parity_checks_actual_stars_and_each_case_not_only_counts():
    cases = [case()]
    score, predictions, _ = evaluate_guard(
        cases, guard_for(cases, [[RecognizerResult("KR_NAME", 0, 3, 1.)]]))
    expected = compact(score)
    assert_baseline_replay(score, predictions, [{(0, 3)}], expected, "test")
    with pytest.raises(ValueError, match="predictions"):
        assert_baseline_replay(score, predictions, [set()], expected, "test")
    with pytest.raises(ValueError, match="fully_covered_names"):
        assert_baseline_replay(score, predictions, [{(0, 3)}],
                               dict(expected, fully_covered_names=0), "test")
