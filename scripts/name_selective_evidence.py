"""Research-only selective additions and removals under legal BIOES paths.

Uncertain baseline anchors remain unchanged. Removing an anchor requires the
joint event that *every* character is outside a name; a low probability for its
exact original name boundary is not evidence of this event. Scores are model
posteriors, not calibrated real-world privacy probabilities or safety guarantees.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
from name_bioes_ablation import TRANSITIONS
from name_span_evidence import select_span_evidence, span_posteriors


def _validate_origin(origin):
    if type(origin) is not int or origin < 0:
        raise ValueError("Expected a nonnegative integer original offset")


def _validate_logits(logits):
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != 5
            or not logits.is_floating_point() or not torch.isfinite(logits).all()):
        raise ValueError("Expected finite floating character logits with five BIOES classes")


def _validate_threshold(value):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("Expected finite probability thresholds in [0, 1]")


def _coordinates(interval, origin, length):
    if not isinstance(interval, (tuple, list)) or len(interval) != 2:
        raise ValueError("Expected intervals as (start, end) pairs")
    start, end = interval
    if (type(start) is not int or type(end) is not int
            or not origin <= start < end <= origin + length):
        raise ValueError("Expected nonempty integer intervals within the text")
    return start, end


def _validated_anchors(anchors, origin, length):
    try:
        anchors = list(anchors)
    except TypeError as exc:
        raise ValueError("Expected anchor (start, end, confidence) triples") from exc
    for anchor in anchors:
        if not isinstance(anchor, (tuple, list)) or len(anchor) != 3:
            raise ValueError("Expected anchor (start, end, confidence) triples")
        _coordinates(anchor[:2], origin, length)
        _validate_threshold(anchor[2])
    anchors = [tuple(anchor) for anchor in anchors]
    ordered = sorted(anchors)
    if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:], strict=False)):
        raise ValueError("Original anchors must not overlap or repeat")
    return anchors


def all_o_posteriors(logits, intervals, *, origin=0):
    """Return P(Y[start:end] is entirely O), in the supplied interval order.

    Intervals use absolute half-open character coordinates. Overlapping and
    repeated queries are valid, and each retains its own output position. All
    legal complete BIOES paths contribute, including context outside the query.
    These are joint events, not products of individual character marginals.

    Forward/backward and O-emission prefix sums take O(text length + queries)
    time and storage. Arithmetic uses float64 log space; output is float64 for
    float64 logits, otherwise float32, on the input device. Inputs are unchanged.
    """
    _validate_logits(logits)
    _validate_origin(origin)
    try:
        intervals = list(intervals)
    except TypeError as exc:
        raise ValueError("Expected an iterable of (start, end) intervals") from exc
    intervals = [_coordinates(interval, origin, len(logits)) for interval in intervals]
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    if not intervals:
        return torch.empty(0, dtype=dtype, device=logits.device)

    unary = logits.double()
    # Per-character constants cancel from each complete path's normalized score.
    unary = unary - unary.max(-1, keepdim=True).values
    transition = unary.new_full((5, 5), -torch.inf)
    for previous, following in enumerate(TRANSITIONS):
        transition[previous, list(following)] = 0.
    start = unary.new_tensor([0., 0., -torch.inf, -torch.inf, 0.])
    end = unary.new_tensor([0., -torch.inf, -torch.inf, 0., 0.])
    forward = [unary[0] + start]
    for row in unary[1:]:
        forward.append(row + (forward[-1][:, None] + transition).logsumexp(0))
    backward = [end]
    for position in range(len(unary) - 2, -1, -1):
        following = unary[position + 1] + backward[-1]
        backward.append((transition + following[None, :]).logsumexp(1))
    forward = torch.stack(forward)
    backward = torch.stack(backward[::-1])
    log_partition = (forward[-1] + end).logsumexp(0)

    # O can follow only O, E, or S, and every internal transition is O -> O.
    prefix = torch.cat((unary.new_zeros(1), forward[:-1, [0, 3, 4]].logsumexp(1)))
    outside_sum = torch.cat((unary.new_zeros(1), unary[:, 0].cumsum(0)))
    starts = torch.tensor([s - origin for s, _ in intervals], device=logits.device)
    ends = torch.tensor([e - origin for _, e in intervals], device=logits.device)
    event = prefix[starts] + outside_sum[ends] - outside_sum[starts] + backward[ends - 1, 0]
    return (event - log_partition).exp().clamp(0., 1.).to(dtype=dtype)


def select_selective_evidence(posteriors, outside_posteriors, addition_threshold,
                              removal_threshold, anchors, *, origin=0):
    """Apply a policy to cached span and anchor-outside event probabilities.

    ``posteriors`` is the matrix from ``span_posteriors``. ``outside_posteriors``
    is a floating vector aligned to the ORIGINAL supplied anchor order, from
    ``all_o_posteriors``. Original anchors must be nonoverlapping and unique,
    even when the policy would remove every anchor. Retained scores are unchanged.

    Remove iff P(all O) > removal_threshold. Then select nonoverlapping new
    spans by the existing maximum sum(P(span) - addition_threshold) policy,
    blocking proposals that overlap a retained anchor. Threshold 1 disables
    its action exactly. Returned triples use absolute coordinates and are sorted.
    """
    _validate_origin(origin)
    _validate_threshold(addition_threshold)
    _validate_threshold(removal_threshold)
    if not isinstance(posteriors, torch.Tensor) or posteriors.ndim != 2:
        raise ValueError("Expected a matrix of complete-span probabilities")
    anchors = _validated_anchors(anchors, origin, len(posteriors))
    if (not isinstance(outside_posteriors, torch.Tensor) or outside_posteriors.ndim != 1
            or len(outside_posteriors) != len(anchors)
            or not outside_posteriors.is_floating_point()
            or not torch.isfinite(outside_posteriors).all()
            or (outside_posteriors < 0).any() or (outside_posteriors > 1).any()):
        raise ValueError("Expected one finite outside probability in [0, 1] per anchor")
    kept = [anchor for anchor, probability in zip(anchors, outside_posteriors.tolist(), strict=True)
            if probability <= removal_threshold]
    return select_span_evidence(posteriors, addition_threshold, origin=origin, anchors=kept)


def selective_spans(logits, addition_threshold, removal_threshold, anchors, *, origin=0):
    """Compute event evidence and selectively change baseline anchor triples.

    This is a research decoder, without model loading or runtime promotion.
    Removal uses the whole outside event; addition uses complete name events.
    ``origin`` maps this full set of character logits to absolute coordinates.
    """
    _validate_logits(logits)
    _validate_origin(origin)
    _validate_threshold(addition_threshold)
    _validate_threshold(removal_threshold)
    anchors = _validated_anchors(anchors, origin, len(logits))
    outside = all_o_posteriors(logits, [(s, e) for s, e, _ in anchors], origin=origin)
    return select_selective_evidence(span_posteriors(logits), outside,
                                     addition_threshold, removal_threshold, anchors,
                                     origin=origin)
