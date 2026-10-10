"""Exact NER gains must not hide worse masking coverage during selection."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from select_name_coverage import selection_rank, spans_from_probabilities
finally:
    sys.path[:] = previous_path


@pytest.mark.parametrize("changes", [
    {"fully_covered_names": 99, "f1": .99},
    {"unnecessary_masked_characters": 21},
    {"negative_false_positive_sentences": 6},
    {"f1": .79},
])
def test_selection_rejects_worse_privacy_or_extra_masking(changes):
    baseline = dict(fully_covered_names=100, f1=.8, unnecessary_masked_characters=20,
                    negative_false_positive_sentences=5)
    assert selection_rank({**baseline, **changes}, baseline) is None
    assert selection_rank(baseline, baseline) == (100, -20, .8)


def test_probability_replay_preserves_inclusive_cutoff_and_original_endpoints():
    assert spans_from_probabilities([.01, .1, .9, .09, .8], .1) == [(1, 3, .5), (4, 5, .8)]
