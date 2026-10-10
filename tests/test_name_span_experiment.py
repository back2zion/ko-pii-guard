"""Research decoder must explore boundaries and choose spans without head priority."""

import itertools
import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from name_span_experiment import SpanCandidate, SpanNameHead, choose_spans, span_loss
finally:
    sys.path[:] = previous_path


def test_two_correct_shorter_spans_can_displace_one_wide_candidate():
    wide = SpanCandidate(0, 7, 4.0)
    left, right = SpanCandidate(0, 2, 3.0), SpanCandidate(3, 7, 3.0)
    for order in itertools.permutations([wide, left, right]):
        assert choose_spans(order) == [left, right]


def test_adjacent_duplicates_negative_scores_and_blocked_address():
    candidates = [
        SpanCandidate(0, 2, 2.0),
        SpanCandidate(2, 4, 3.0),
        SpanCandidate(0, 2, 1.0),
        SpanCandidate(4, 7, -1.0),
        SpanCandidate(8, 12, 20.0),
    ]
    assert choose_spans(candidates, blocked=[(10, 15)]) == candidates[:2]


def test_interval_solution_matches_exhaustive_subset_search():
    import random

    random.seed(7)
    for _ in range(40):
        candidates = [
            SpanCandidate(s, s + random.randint(1, 4), random.random() * 4 - 1) for s in range(8)
        ]
        result = choose_spans(candidates)
        best = 0.0
        for bits in itertools.product([False, True], repeat=len(candidates)):
            subset = sorted(
                (c for c, keep in zip(candidates, bits, strict=True) if keep), key=lambda c: c.start
            )
            if all(a.end <= b.start for a, b in zip(subset, subset[1:], strict=False)):
                best = max(best, sum(c.score for c in subset))
        assert sum(c.score for c in result) == pytest.approx(best)


@pytest.mark.parametrize(
    "candidate", [SpanCandidate(-1, 2, 1), SpanCandidate(2, 2, 1), SpanCandidate(0, 2, math.nan)]
)
def test_invalid_coordinates_or_scores_rejected(candidate):
    with pytest.raises(ValueError):
        choose_spans([candidate])


def test_model_scores_all_boundaries_and_masks_padding_and_backpropagates():
    import torch

    torch.manual_seed(1)
    model = SpanNameHead(
        vocabulary_size=10, hidden_size=8, projection_size=4, char_size=3, recurrent_size=4
    )
    features = torch.randn(2, 5, 8)
    chars = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 0, 0, 0]])
    positions = torch.zeros(2, 5, 2)
    lengths = torch.tensor([5, 2])
    logits, valid = model(features, chars, positions, lengths, max_span_width=3)
    assert logits.shape == valid.shape == (2, 5, 3)
    assert valid.sum(1).tolist() == [[5, 4, 3], [2, 1, 0]]
    # One-character names and token-internal endings are real candidates.
    assert valid[0, 1, 0] and valid[0, 1, 1]
    targets = torch.zeros_like(logits)
    targets[0, 1, 1] = 1
    loss = span_loss(logits, valid, targets, positive_weight=3.0)
    loss.backward()
    assert model.scorer[-1].weight.grad.abs().sum() > 0
    altered = logits.detach().clone()
    altered[~valid] = 10000
    assert span_loss(altered, valid, targets, positive_weight=3.0) == pytest.approx(loss.item())
