"""Research name-coverage decoder marginalizing all legal BIOES sequences.

Scores are posteriors under the supplied unary logits and legal transitions,
not calibrated privacy probabilities. Contiguous selected characters may merge
adjacent names; this decoder targets character coverage rather than exact spans.
"""

from __future__ import annotations

import math

import torch
from name_bioes_ablation import TRANSITIONS


def name_probabilities(logits):
    """Return P(B or I or E or S) at each original character position.

Forward-backward sums all paths with legal BIOES transitions, start O/B/S, and
end O/E/S. Computation uses normalized double-precision log space; output keeps
float32/float64 input precision, promoting float16/bfloat16 outputs to float32.
"""
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != 5
            or not logits.is_floating_point() or not torch.isfinite(logits).all()):
        raise ValueError("Expected finite floating character logits with five BIOES classes")
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    if not len(logits):
        return torch.empty(0, dtype=dtype, device=logits.device)
    unary = logits.double()
    # Per-position constants cancel from every legal path's normalized weight.
    unary = unary - unary.max(-1, keepdim=True).values
    transition = unary.new_full((5, 5), -torch.inf)
    for previous, following in enumerate(TRANSITIONS):
        transition[previous, list(following)] = 0.
    start = unary.new_tensor([0., 0., -torch.inf, -torch.inf, 0.])
    end = unary.new_tensor([0., -torch.inf, -torch.inf, 0., 0.])

    def normalize(values):
        return values - values.logsumexp(0)

    forward = [normalize(unary[0] + start)]
    for row in unary[1:]:
        forward.append(normalize(row + (forward[-1][:, None] + transition).logsumexp(0)))
    backward = [normalize(end)]
    for position in range(len(unary) - 2, -1, -1):
        following = unary[position + 1] + backward[-1]
        backward.append(normalize((transition + following[None]).logsumexp(1)))
    posterior = (torch.stack(forward) + torch.stack(backward[::-1])).softmax(-1)
    return posterior[:, 1:].sum(-1).clamp(0., 1.).to(dtype=dtype)


def coverage_spans(logits, threshold, origin=0):
    """Return contiguous posterior >= threshold spans with mean confidence."""
    if (not math.isfinite(threshold) or not 0 <= threshold <= 1
            or not isinstance(origin, int) or origin < 0):
        raise ValueError("Expected a probability threshold and nonnegative original offset")
    probabilities = name_probabilities(logits).tolist()
    spans, start = [], None
    for index, score in enumerate([*probabilities, -1.]):
        if score >= threshold:
            if start is None:
                start = index
        elif start is not None:
            spans.append((origin + start, origin + index,
                          sum(probabilities[start:index]) / (index - start)))
            start = None
    return spans
