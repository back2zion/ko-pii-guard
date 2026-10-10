"""Posterior coverage must marginalize every legal BIOES path exactly."""

import itertools
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from name_coverage_decoding import coverage_spans, name_probabilities
finally:
    sys.path[:] = previous_path


def exhaustive_probabilities(logits):
    transitions = ((0, 1, 4), (2, 3), (2, 3), (0, 1, 4), (0, 1, 4))
    paths = [path for path in itertools.product(range(5), repeat=len(logits))
             if path[0] in (0, 1, 4) and path[-1] in (0, 3, 4)
             and all(b in transitions[a] for a, b in zip(path, path[1:], strict=False))]
    scores = torch.tensor([sum(float(logits[i, tag]) for i, tag in enumerate(path))
                           for path in paths], dtype=torch.float64)
    weights = scores.softmax(0)
    return torch.tensor([sum(float(weights[j]) for j, path in enumerate(paths) if path[i] != 0)
                         for i in range(len(logits))], dtype=torch.float64)


@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_marginal_probability_matches_exhaustive_legal_paths(length):
    generator = torch.Generator().manual_seed(928 + length)
    logits = torch.randn(length, 5, generator=generator, dtype=torch.float64) * 2
    assert torch.allclose(name_probabilities(logits), exhaustive_probabilities(logits),
                          rtol=1e-12, atol=1e-12)


def test_invalid_opening_tags_cannot_inflate_single_character_name_probability():
    logits = torch.tensor([[0., 100., 200., 300., 0.]])
    assert name_probabilities(logits).item() == pytest.approx(0.5)


def test_contiguous_coverage_merges_adjacent_names_and_preserves_origin():
    logits = torch.full((4, 5), -20.)
    logits[:, 0] = 20
    logits[1:3, 0] = -20
    logits[1:3, 4] = 20
    (start, end, score), = coverage_spans(logits, threshold=0.9, origin=7)
    assert (start, end) == (8, 10)
    assert score == pytest.approx(1.)


def test_extreme_logits_and_per_character_constant_shifts_are_stable():
    logits = torch.tensor([[1000., -1000., -1000., -1000., 999.],
                           [999., -1000., -1000., -1000., 1000.]])
    assert torch.isfinite(name_probabilities(logits)).all()
    assert torch.allclose(name_probabilities(logits),
                          name_probabilities(logits + torch.tensor([[5000.], [-4000.]])))


def test_empty_input_and_invalid_tensor_shape_dtype_or_values():
    assert name_probabilities(torch.empty(0, 5)).shape == (0,)
    assert coverage_spans(torch.empty(0, 5), 0.5) == []
    for logits in (torch.empty(2, 4), torch.empty(5), torch.zeros(1, 5, dtype=torch.long),
                   torch.tensor([[0., 0., float("nan"), 0., 0.]]),
                   torch.tensor([[0., 0., float("inf"), 0., 0.]])):
        with pytest.raises(ValueError, match="finite floating"):
            name_probabilities(logits)


@pytest.mark.parametrize("threshold,origin", [(-0.1, 0), (1.1, 0), (float("nan"), 0),
                                            (0.5, -1), (0.5, 1.5)])
def test_invalid_threshold_or_origin_is_rejected(threshold, origin):
    with pytest.raises(ValueError, match="threshold.*offset"):
        coverage_spans(torch.zeros(2, 5), threshold=threshold, origin=origin)
