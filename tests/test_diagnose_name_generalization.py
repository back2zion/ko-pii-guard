"""Coverage diagnostics distinguish threshold loss from missing decoder paths."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from diagnose_name_generalization import attribute, covered, reconstruct
finally:
    sys.path[:] = previous_path


def test_reconstruction_uses_correct_gold_plus_false_predictions_and_checks_totals():
    cases = [dict(id="a", text="가나다라", expected=[
        dict(entity="KR_NAME", start=0, end=2), dict(entity="KR_NAME", start=3, end=4),
    ])]
    score = dict(tp=1, fp=1, fn=1,
                 errors=[dict(id="a", missed=[[0, 2]], false=[[0, 1]])])
    assert reconstruct(cases, score) == [{(0, 1), (3, 4)}]
    with pytest.raises(ValueError, match="reconstruction"):
        reconstruct(cases, {**score, "tp": 2})


def test_loss_attribution_only_counts_previously_fully_covered_names():
    cases = [dict(id="a", text="가나다라마바사아자차카", expected=[
        dict(entity="KR_NAME", start=start, end=end)
        for start, end in [(0, 2), (3, 5), (6, 8), (9, 11)]
    ])]
    result = attribute(cases, [{(0, 2), (3, 5), (6, 8)}], [set()], [{(0, 2), (3, 4)}])
    assert result["counts"] == dict(confidence_threshold=1, path_boundary=1, no_path_proposal=1)
    assert result["examples"]["confidence_threshold"] == [("a", 0, 2)]
    assert result["examples"]["path_boundary"] == [("a", 3, 5)]
    assert result["examples"]["no_path_proposal"] == [("a", 6, 8)]


def test_coverage_uses_union_and_requires_every_character():
    assert covered(1, 5, {(0, 2), (2, 4), (4, 6)})
    assert not covered(1, 5, {(0, 2), (3, 6)})
