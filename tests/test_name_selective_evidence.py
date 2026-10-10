"""Selective changes require joint outside evidence and preserve uncertain anchors."""

import itertools
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from name_selective_evidence import (
        all_o_posteriors,
        select_selective_evidence,
        selective_spans,
    )
    from name_span_evidence import span_posteriors
finally:
    sys.path[:] = previous_path


def exhaustive_outside_events(logits, intervals):
    transitions = ((0, 1, 4), (2, 3), (2, 3), (0, 1, 4), (0, 1, 4))
    paths = [path for path in itertools.product(range(5), repeat=len(logits))
             if path[0] in (0, 1, 4) and path[-1] in (0, 3, 4)
             and all(b in transitions[a] for a, b in zip(path, path[1:], strict=False))]
    scores = torch.tensor([sum(float(logits[i, tag]) for i, tag in enumerate(path))
                           for path in paths], dtype=torch.float64)
    weights = scores.softmax(0)
    return torch.tensor([sum(float(weight) for path, weight in zip(paths, weights, strict=True)
                             if all(tag == 0 for tag in path[start:end]))
                         for start, end in intervals], dtype=torch.float64)


@pytest.mark.parametrize("length", range(1, 6))
def test_every_interval_matches_exhaustive_legal_paths(length):
    logits = torch.randn(length, 5, dtype=torch.float64,
                         generator=torch.Generator().manual_seed(671 + length)) * 2
    intervals = [(start, end) for start in range(length)
                 for end in range(start + 1, length + 1)]
    actual = all_o_posteriors(logits, intervals)
    assert torch.allclose(actual, exhaustive_outside_events(logits, intervals),
                          atol=1e-12, rtol=1e-12)


def test_joint_outside_event_is_not_product_of_character_marginals():
    # Five equiprobable legal paths: OO, OS, BE, SO, SS. The BE alternative
    # couples the characters, so P(OO)=1/5, whereas P(O at 0)*P(O at 1)=4/25.
    probabilities = all_o_posteriors(torch.zeros(2, 5, dtype=torch.float64),
                                    [(0, 1), (1, 2), (0, 2)])
    assert probabilities.tolist() == pytest.approx([0.4, 0.4, 0.2])
    assert abs(probabilities[2] - probabilities[0] * probabilities[1]) > 0.03


def test_illegal_singleton_states_cannot_suppress_outside_probability():
    logits = torch.tensor([[0., 100., 200., 300., 0.]], dtype=torch.float64)
    assert all_o_posteriors(logits, [(0, 1)]).item() == pytest.approx(0.5)


def test_query_order_repetition_overlap_and_absolute_offsets_are_preserved():
    logits = torch.randn(3, 5, dtype=torch.float64,
                         generator=torch.Generator().manual_seed(89))
    intervals = [(1, 3), (0, 2), (1, 3), (0, 3)]
    shifted = [(start + 11, end + 11) for start, end in intervals]
    actual = all_o_posteriors(logits, shifted, origin=11)
    assert torch.allclose(actual, exhaustive_outside_events(logits, intervals))
    assert actual[0] == actual[2]


def confident_name_outside_name():
    logits = torch.full((4, 5), -20.)
    logits[range(4), [4, 0, 0, 4]] = 20.
    return logits


@pytest.mark.parametrize("addition,removal,expected", [
    (0.9, 0.99, [(10, 11), (13, 14)]),
    (0.9, 1., [(10, 11), (11, 13), (13, 14)]),
    (1., 0.99, [(10, 11)]),
    (1., 1., [(10, 11), (11, 13)]),
])
def test_bidirectional_addition_only_removal_only_and_no_change(addition, removal, expected):
    logits = confident_name_outside_name()
    original = logits.clone()
    # Deliberately unsorted input: outside probabilities must stay paired with
    # the original anchors, while final output follows document order.
    anchors = [(11, 13, 0.72), (10, 11, 0.83)]
    result = selective_spans(logits, addition, removal, anchors, origin=10)
    assert [(start, end) for start, end, _ in result] == expected
    assert result[0] == (10, 11, 0.83)  # True single-character name is preserved.
    if removal == 1.:
        assert result[1] == (11, 13, 0.72)
    assert anchors == [(11, 13, 0.72), (10, 11, 0.83)]
    assert torch.equal(logits, original)


def test_removal_and_addition_use_strict_thresholds_and_uncertainty_keeps_anchor():
    logits = torch.zeros(1, 5, dtype=torch.float64)
    anchor = [(0, 1, 0.987)]
    assert selective_spans(logits, 1., 0.5, anchor) == anchor
    assert selective_spans(logits, 1., 0.49, anchor) == []
    assert selective_spans(logits, 0.5, 1., []) == []
    assert selective_spans(logits, 0.49, 1., []) == [(0, 1, 0.5)]
    assert selective_spans(logits, 0.99, 0.99, anchor) == anchor


def test_kept_anchor_blocks_overlapping_name_proposals():
    logits = torch.full((3, 5), -20.)
    logits[range(3), [1, 2, 3]] = 20.
    anchor = [(1, 2, 0.9)]
    assert selective_spans(logits, 0.5, 0.99, anchor) == anchor


def test_after_removal_global_nonoverlap_selection_maximizes_expected_surplus():
    logits = torch.randn(3, 5, dtype=torch.float64,
                         generator=torch.Generator().manual_seed(290))
    threshold = 0.04
    posteriors = span_posteriors(logits)
    proposals = [(start, start + width + 1, float(score))
                 for start, row in enumerate(posteriors)
                 for width, score in enumerate(row) if score > threshold]
    optimum = 0.
    for mask in itertools.product((False, True), repeat=len(proposals)):
        chosen = sorted(span for span, keep in zip(proposals, mask, strict=True) if keep)
        if all(a[1] <= b[0] for a, b in zip(chosen, chosen[1:], strict=False)):
            optimum = max(optimum, sum(score - threshold for _, _, score in chosen))
    result = selective_spans(logits, threshold, 0., [(0, 1, 0.8)])
    assert sum(score - threshold for _, _, score in result) == pytest.approx(optimum)


def test_cached_selection_matches_live_and_does_not_change_evidence():
    logits = torch.randn(5, 5, dtype=torch.float64,
                         generator=torch.Generator().manual_seed(231))
    anchors = [(6, 8, 0.92), (3, 4, 0.85)]
    span_probabilities = span_posteriors(logits)
    outside = all_o_posteriors(logits, [(s, e) for s, e, _ in anchors], origin=3)
    originals = span_probabilities.clone(), outside.clone()
    for addition, removal in itertools.product((0., 0.25, 0.9, 1.), (0., 0.5, 0.99, 1.)):
        assert select_selective_evidence(span_probabilities, outside, addition, removal,
                                         anchors, origin=3) == selective_spans(
            logits, addition, removal, anchors, origin=3)
    assert torch.equal(span_probabilities, originals[0])
    assert torch.equal(outside, originals[1])


def test_json_list_and_tuple_anchors_can_mix_and_adjacent_anchors_remain_distinct():
    anchors = [(1, 2, 0.8), [0, 1, 0.9]]
    assert selective_spans(torch.zeros(2, 5), 1., 1., anchors) == [
        (0, 1, 0.9), (1, 2, 0.8)]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_finite_numerics_dtype_and_empty_inputs(dtype):
    logits = torch.tensor([[1000., -1000., -1000., -1000., 999.],
                           [999., -1000., -1000., -1000., 1000.]], dtype=dtype)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    result = all_o_posteriors(logits, [(0, 1), (0, 2)])
    assert result.dtype == expected_dtype and result.device == logits.device
    assert torch.isfinite(result).all() and ((0 <= result) & (result <= 1)).all()
    assert all_o_posteriors(torch.empty(0, 5, dtype=dtype), []).shape == (0,)
    assert all_o_posteriors(logits, []).shape == (0,)
    assert selective_spans(torch.empty(0, 5, dtype=dtype), 0.5, 0.9, []) == []


def test_outside_probabilities_are_invariant_to_per_character_constants():
    logits = torch.randn(5, 5, dtype=torch.float64,
                         generator=torch.Generator().manual_seed(725))
    shifts = torch.tensor([1e6, -1e6, 1e6, -1e6, 1e6])[:, None]
    intervals = [(0, 5), (1, 2), (1, 4)]
    assert torch.allclose(all_o_posteriors(logits, intervals),
                          all_o_posteriors(logits + shifts, intervals),
                          atol=1e-10, rtol=1e-10)


def test_invalid_logits_and_intervals_are_rejected_even_with_no_requested_changes():
    for logits in (None, torch.zeros(5), torch.zeros(2, 4), torch.ones(1, 5, dtype=torch.long),
                   torch.full((1, 5), math.nan), torch.full((1, 5), math.inf)):
        with pytest.raises(ValueError):
            all_o_posteriors(logits, [])
        with pytest.raises(ValueError):
            selective_spans(logits, 1., 1., [])
    logits = torch.zeros(3, 5)
    for interval in ((0, 0), (2, 1), (-1, 1), (0, 4), (0.5, 2), (False, 1),
                     (0,), (0, 1, 2), None, "01"):
        with pytest.raises(ValueError):
            all_o_posteriors(logits, [interval])
    for origin in (-1, 0.5, True):
        with pytest.raises(ValueError):
            all_o_posteriors(logits, [], origin=origin)


@pytest.mark.parametrize("anchors", [
    [(0, 1, 0.9), (0, 1, 0.9)], [(0, 2, 0.9), (1, 3, 0.9)],
    [(0, 4, 0.9)], [(0, 0, 0.9)], [(0, 1, math.nan)], [(0, 1, 1.1)],
    [(0, 1, -0.1)], [(0, 1)], [(False, 1, 0.9)], [None],
])
def test_invalid_anchors_cannot_be_hidden_by_deletion(anchors):
    logits = torch.full((3, 5), -20.)
    logits[:, 0] = 20.
    with pytest.raises(ValueError):
        selective_spans(logits, 1., 0., anchors)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, math.nan, math.inf, "0.9", True])
def test_invalid_thresholds_are_rejected(threshold):
    for addition, removal in ((threshold, 1.), (1., threshold)):
        with pytest.raises(ValueError):
            selective_spans(torch.zeros(1, 5), addition, removal, [])


def test_cached_selection_rejects_missing_or_invalid_event_probabilities():
    spans = torch.tensor([[0.5]])
    for outside in (torch.empty(0), torch.zeros(1, 1), torch.ones(1, dtype=torch.long),
                    torch.tensor([math.nan]), torch.tensor([-0.1]), torch.tensor([1.1])):
        with pytest.raises(ValueError):
            select_selective_evidence(spans, outside, 1., 0., [(0, 1, 0.8)])
    with pytest.raises(ValueError):
        select_selective_evidence(torch.ones(2, 2), torch.ones(0), 1., 1., [])
