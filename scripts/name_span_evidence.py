"""Research decoder using complete-name events rather than character tails.

Probabilities are exact marginals under the supplied logits and legal BIOES
paths. They are not empirically calibrated probabilities of a real name. A
complete span must be S or B I* E; incompatible boundary alternatives cannot
be combined into a larger event. No lexical or minimum-length rules are used.
"""

from __future__ import annotations

import math

import torch
from name_bioes_ablation import TRANSITIONS


def span_posteriors(logits, max_span_width=None):
    """Return P(the complete span [s, s+w) occurs), indexed [s, w-1].

    The result has shape [characters, min(characters, max_span_width)], or
    [characters, characters] when width is omitted. Impossible end positions
    contain zero. A width limit omits candidates without changing the sequence
    distribution or renormalizing the remaining probabilities. Forward-backward
    takes O(n) time; candidate computation and storage take O(n * width).

    Calculations use float64 log space. Output preserves float64 inputs and
    otherwise uses float32, on the input device. Empty input returns [0, 0].
    """
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != 5
            or not logits.is_floating_point() or not torch.isfinite(logits).all()):
        raise ValueError("Expected finite floating character logits with five BIOES classes")
    if max_span_width is not None and (
            type(max_span_width) is not int or max_span_width < 1):
        raise ValueError("Expected a positive integer maximum span width")
    n = len(logits)
    width = n if max_span_width is None else min(n, max_span_width)
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    if not n:
        return torch.empty((0, 0), dtype=dtype, device=logits.device)

    unary = logits.double()
    # Every full path has one emission at each position, so these constants cancel.
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
    for position in range(n - 2, -1, -1):
        following = unary[position + 1] + backward[-1]
        backward.append((transition + following[None, :]).logsumexp(1))
    forward = torch.stack(forward)
    backward = torch.stack(backward[::-1])
    log_partition = (forward[-1] + end).logsumexp(0)

    # Both B and S can follow only O, E, or S. Before position zero there is
    # one empty prefix with log weight zero. S/E suffixes enforce a true end.
    prefix = torch.cat((unary.new_zeros(1), forward[:-1, [0, 3, 4]].logsumexp(1)))
    inside_prefix = torch.cat((unary.new_zeros(1), unary[:, 2].cumsum(0)))
    starts = torch.arange(n, device=logits.device)[:, None]
    widths = torch.arange(1, width + 1, device=logits.device)[None, :]
    last = (starts + widths - 1).clamp(max=n - 1)
    inside = inside_prefix[last] - inside_prefix[(starts + 1).clamp(max=n)]
    multiple = unary[:, 1, None] + inside + unary[last, 3] + backward[last, 3]
    singleton = unary[:, 4] + backward[:, 4]
    event = torch.where(widths == 1, singleton[:, None], multiple)
    log_probability = prefix[:, None] + event - log_partition
    log_probability = log_probability.masked_fill(starts + widths > n, -torch.inf)
    return log_probability.exp().clamp(0., 1.).to(dtype=dtype)


def select_span_evidence(posteriors, threshold, origin=0, anchors=()):
    """Select nonoverlapping cached spans maximizing sum(probability - threshold).

    Candidates require probability strictly above the cutoff. This is a span
    existence decision with equal per-span benefit/cost, not a guarantee of
    privacy coverage or real-world calibration. Dynamic programming takes
    O(n * width) time and O(n) additional storage, excluding the input copy.

    Optional anchors are absolute-coordinate (start, end, confidence) triples.
    They are preserved unchanged and block overlapping proposals. Their scores
    remain upstream confidence values, not marginals from this model. Anchoring
    preserves supplied coverage but also preserves supplied false positives.
    Invalid end-position cells must be zero; overlapping anchors are rejected.
    """
    if (not isinstance(posteriors, torch.Tensor) or posteriors.ndim != 2
            or not posteriors.is_floating_point() or not torch.isfinite(posteriors).all()
            or (posteriors < 0).any() or (posteriors > 1).any()):
        raise ValueError("Expected a finite matrix of span probabilities in [0, 1]")
    if (not math.isfinite(threshold) or not 0 <= threshold <= 1
            or type(origin) is not int or origin < 0):
        raise ValueError("Expected a probability threshold and nonnegative original offset")
    n, width = posteriors.shape
    if width > n or (n and not width):
        raise ValueError("Expected span matrix width between one and text length")
    rows = posteriors.detach().cpu().tolist()
    if any(any(row[n - start:]) for start, row in enumerate(rows)):
        raise ValueError("Span probabilities outside text bounds must be zero")

    protected = []
    for anchor in anchors:
        if len(anchor) != 3:
            raise ValueError("Expected anchor (start, end, confidence) triples")
        start, end, confidence = anchor
        if (type(start) is not int or type(end) is not int
                or not origin <= start < end <= origin + n
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("Expected finite anchor confidence and coordinates within the text")
        protected.append((start, end, confidence))
    protected.sort()
    if any(a[1] > b[0] for a, b in zip(protected, protected[1:], strict=False)):
        raise ValueError("Protected anchors must not overlap")
    blocked = [0] * n
    for start, end, _ in protected:
        blocked[start - origin:end - origin] = [1] * (end - start)
    occupied = [0]
    for value in blocked:
        occupied.append(occupied[-1] + value)

    best = [0.] * (n + 1)
    chosen_start = [None] * (n + 1)
    for end in range(1, n + 1):
        best[end] = best[end - 1]
        for start in range(max(0, end - width), end):
            probability = rows[start][end - start - 1]
            if probability <= threshold or occupied[end] != occupied[start]:
                continue
            utility = best[start] + probability - threshold
            if utility > best[end]:
                best[end] = utility
                chosen_start[end] = start
    selected, end = [], n
    while end:
        start = chosen_start[end]
        if start is None:
            end -= 1
        else:
            selected.append((origin + start, origin + end, rows[start][end - start - 1]))
            end = start
    return sorted([*selected, *protected])


def evidence_spans(logits, threshold, origin=0, max_span_width=None, anchors=()):
    """Compute exact whole-span posteriors and select nonoverlapping evidence."""
    return select_span_evidence(
        span_posteriors(logits, max_span_width=max_span_width),
        threshold=threshold, origin=origin, anchors=anchors,
    )
