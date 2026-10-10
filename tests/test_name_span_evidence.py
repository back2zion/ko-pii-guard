"""Whole-span event probabilities match exhaustive legal BIOES path sums."""

import itertools
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from name_bioes_ablation import TRANSITIONS
    from name_span_evidence import evidence_spans, select_span_evidence, span_posteriors
finally:
    sys.path[:] = previous_path


def enumerate_posteriors(logits):
    paths, weights = [], []
    for path in itertools.product(range(5), repeat=len(logits)):
        if path[0] not in (0, 1, 4) or path[-1] not in (0, 3, 4):
            continue
        if any(b not in TRANSITIONS[a] for a, b in zip(path, path[1:], strict=False)):
            continue
        paths.append(path)
        weights.append(sum(logits[i, label] for i, label in enumerate(path)))
    probabilities = torch.stack(weights).softmax(0)
    expected = torch.zeros(len(logits), len(logits), dtype=torch.float64)
    for path, probability in zip(paths, probabilities, strict=True):
        start = None
        for i, label in enumerate(path):
            if label == 1:
                start = i
            elif label == 3:
                expected[start, i - start] += probability
                start = None
            elif label == 4:
                expected[i, 0] += probability
    return expected


@pytest.mark.parametrize("length", range(1, 6))
def test_every_span_marginal_matches_exhaustive_legal_sequence_enumeration(length):
    generator = torch.Generator().manual_seed(300 + length)
    logits = torch.randn(length, 5, dtype=torch.float64, generator=generator)
    assert torch.allclose(span_posteriors(logits), enumerate_posteriors(logits),
                          atol=1e-12, rtol=1e-12)


def test_width_cap_omits_long_candidates_without_renormalizing_shorter_events():
    logits = torch.randn(5, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(9))
    assert torch.allclose(span_posteriors(logits, max_span_width=2),
                          span_posteriors(logits)[:, :2])
    assert span_posteriors(logits, max_span_width=2)[-1, 1] == 0


def test_single_character_true_name_is_retained_but_outside_tail_is_rejected():
    name = torch.tensor([[0., -10., -10., -10., 8.]])
    outside = torch.tensor([[8., -10., -10., -10., 0.]])
    assert [(s, e) for s, e, _ in evidence_spans(name, threshold=0.5, origin=12)] == [(12, 13)]
    assert evidence_spans(outside, threshold=0.5) == []


def test_incompatible_boundary_alternatives_do_not_become_one_complete_name():
    # Almost equal probability for B,E,O and O,B,E. Individual characters can
    # all look name-like, while the encompassing B,I,E span has negligible mass.
    logits = torch.tensor([[0., 0., -20., -20., -20.],
                           [-20., 0., -20., 0., -20.],
                           [0., -20., -20., 0., -20.]], dtype=torch.float64)
    posteriors = span_posteriors(logits)
    assert posteriors[0, 1] == pytest.approx(0.5, abs=1e-7)
    assert posteriors[1, 1] == pytest.approx(0.5, abs=1e-7)
    assert posteriors[0, 2] < 1e-8
    spans = evidence_spans(logits, threshold=0.4)
    assert len(spans) == 1
    assert (spans[0][0], spans[0][1]) in ((0, 2), (1, 3))


def test_nonoverlapping_selection_matches_exhaustive_expected_surplus():
    logits = torch.randn(4, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(28))
    threshold = 0.04
    probabilities = span_posteriors(logits)
    candidates = [(s, s + w + 1, float(p)) for s, row in enumerate(probabilities)
                  for w, p in enumerate(row) if p > threshold]
    best = 0.
    for keep in itertools.product((False, True), repeat=len(candidates)):
        chosen = sorted(candidate for candidate, included in zip(candidates, keep, strict=True)
                        if included)
        if all(a[1] <= b[0] for a, b in zip(chosen, chosen[1:], strict=False)):
            best = max(best, sum(p - threshold for _, _, p in chosen))
    actual = evidence_spans(logits, threshold)
    assert sum(p - threshold for _, _, p in actual) == pytest.approx(best)


def test_protected_anchor_is_unchanged_and_blocks_only_overlapping_proposals():
    logits = torch.full((4, 5), -20.)
    logits[range(4), [1, 3, 0, 4]] = 20.
    anchor = (11, 13, 0.97)
    result = evidence_spans(logits, threshold=0.5, origin=10, anchors=[anchor])
    assert result[0] == anchor
    assert [(s, e) for s, e, _ in result] == [(11, 13), (13, 14)]


def test_numerics_are_finite_and_invariant_to_large_per_position_constants():
    logits = torch.randn(4, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    shifts = torch.tensor([1e6, -1e6, 1e6, -1e6], dtype=torch.float64)[:, None]
    assert torch.allclose(span_posteriors(logits), span_posteriors(logits + shifts),
                          atol=1e-10, rtol=1e-10)
    assert torch.isfinite(span_posteriors(torch.tensor([[1e30, -1e30, 0., 0., 1e30]]))).all()
    assert span_posteriors(torch.empty(0, 5)).shape == (0, 0)
    assert evidence_spans(torch.empty(0, 5), 0.5) == []


def test_invalid_logits_width_threshold_and_anchors_are_rejected():
    with pytest.raises(ValueError):
        span_posteriors(torch.full((2, 5), math.nan))
    with pytest.raises(ValueError):
        span_posteriors(torch.zeros(2, 5), max_span_width=0)
    with pytest.raises(ValueError):
        evidence_spans(torch.zeros(2, 5), threshold=math.nan)
    with pytest.raises(ValueError):
        evidence_spans(torch.zeros(3, 5), 0.5, anchors=[(0, 2, 0.9), (1, 3, 0.8)])
    with pytest.raises(ValueError):
        evidence_spans(torch.zeros(3, 5), 0.5, anchors=[(0, 4, 0.9)])


def test_cached_selector_reuses_probabilities_across_thresholds_and_anchor_modes():
    logits = torch.randn(6, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(28))
    posterior = span_posteriors(logits, max_span_width=3)
    original = posterior.clone()
    for threshold, anchors in itertools.product((0., 0.01, 0.25, 0.5, 1.),
                                                ((), ((5, 6, 0.98),))):
        assert select_span_evidence(posterior, threshold, origin=3, anchors=anchors) == (
            evidence_spans(logits, threshold, origin=3, max_span_width=3, anchors=anchors)
        )
    assert torch.equal(posterior, original)


def test_cached_selector_rejects_impossible_end_positions_and_bad_matrix_shapes():
    for posterior in (torch.ones(2, 2), torch.zeros(2, 3), torch.zeros(2, 0),
                      torch.full((2, 1), -0.1), torch.full((2, 1), 1.1)):
        with pytest.raises(ValueError):
            select_span_evidence(posterior, 0.5)


def test_adjacent_single_character_names_remain_distinct_events():
    logits = torch.full((2, 5), -20.)
    logits[:, 4] = 20.
    assert [(s, e) for s, e, _ in evidence_spans(logits, 0.5)] == [(0, 1), (1, 2)]
