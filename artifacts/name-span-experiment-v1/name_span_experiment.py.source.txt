"""Research-only joint character-span scorer and exact interval decoder.

Not used by the package runtime. Scores are logits, not calibrated probabilities.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SpanCandidate:
    start: int
    end: int
    score: float


def choose_spans(candidates, *, margin=0.0, blocked=()):
    """Maximize total positive logit margin; no model/head-order precedence.

    Addresses may be supplied as blocked ranges. Ties deterministically retain
    the earlier solution. Duplicate coordinates contribute only once.
    """
    if not math.isfinite(margin):
        raise ValueError("Margin must be finite")
    for start, end in blocked:
        if not 0 <= start < end:
            raise ValueError("Invalid blocked span")
    unique = {}
    for candidate in candidates:
        if not (0 <= candidate.start < candidate.end and math.isfinite(candidate.score)):
            raise ValueError("Invalid candidate span or score")
        if candidate.score <= margin or any(
            candidate.start < end and start < candidate.end for start, end in blocked
        ):
            continue
        key = candidate.start, candidate.end
        if key not in unique or candidate.score > unique[key].score:
            unique[key] = candidate
    ordered = sorted(unique.values(), key=lambda c: (c.end, c.start))
    ends = [c.end for c in ordered]
    best, take, predecessors = [0.0], [], []
    for i, candidate in enumerate(ordered):
        predecessor = bisect.bisect_right(ends, candidate.start, hi=i)
        score = best[predecessor] + candidate.score - margin
        selected = score > best[-1]
        best.append(score if selected else best[-1])
        take.append(selected)
        predecessors.append(predecessor)
    result, i = [], len(ordered)
    while i:
        if take[i - 1]:
            result.append(ordered[i - 1])
            i = predecessors[i - 1]
        else:
            i -= 1
    return list(reversed(result))


class SpanNameHead(nn.Module):
    """One joint logit for every character start/width, with ordered endpoints."""

    def __init__(
        self,
        vocabulary_size,
        hidden_size=768,
        projection_size=64,
        char_size=16,
        recurrent_size=64,
        max_width=32,
    ):
        super().__init__()
        self.projection = nn.Linear(hidden_size, projection_size)
        self.characters = nn.Embedding(vocabulary_size, char_size, padding_idx=0)
        self.sequence = nn.LSTM(
            projection_size + char_size + 2, recurrent_size, batch_first=True, bidirectional=True
        )
        self.widths = nn.Embedding(max_width, 16)
        self.scorer = nn.Sequential(
            nn.Linear(recurrent_size * 6 + 16, 64), nn.GELU(), nn.Linear(64, 1)
        )
        nn.init.constant_(self.scorer[-1].bias, -3.0)

    def forward(self, features, characters, positions, lengths, *, max_span_width):
        if not 1 <= max_span_width <= self.widths.num_embeddings:
            raise ValueError("Unsupported maximum span width")
        if (lengths <= 0).any() or (lengths > features.shape[1]).any():
            raise ValueError("Invalid sequence lengths")
        inputs = torch.cat(
            (self.projection(features.float()), self.characters(characters), positions.float()),
            dim=-1,
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            inputs, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        sequence, _ = self.sequence(packed)
        sequence, _ = nn.utils.rnn.pad_packed_sequence(
            sequence, batch_first=True, total_length=features.shape[1]
        )
        size, width = sequence.shape[1], max_span_width
        starts = torch.arange(size, device=sequence.device)[:, None].expand(size, width)
        widths = torch.arange(width, device=sequence.device)[None, :].expand(size, width)
        ends = starts + widths
        valid = ends[None, :, :] < lengths.to(sequence.device)[:, None, None]
        clamped = ends.clamp(max=size - 1)
        prefix = torch.cat(
            (sequence.new_zeros(sequence.shape[0], 1, sequence.shape[-1]), sequence.cumsum(1)),
            dim=1,
        )
        inside = (prefix[:, clamped + 1] - prefix[:, starts]) / (widths + 1)[None, :, :, None]
        values = torch.cat(
            (
                sequence[:, starts],
                sequence[:, clamped],
                inside,
                self.widths(widths)[None].expand(sequence.shape[0], -1, -1, -1),
            ),
            dim=-1,
        )
        return self.scorer(values).squeeze(-1), valid


def span_loss(logits, valid, targets, *, positive_weight):
    if positive_weight <= 0 or not torch.isfinite(torch.tensor(positive_weight)):
        raise ValueError("Positive weight must be finite and positive")
    if not valid.any():
        raise ValueError("No valid span candidates")
    return nn.functional.binary_cross_entropy_with_logits(
        logits[valid], targets[valid], pos_weight=logits.new_tensor(positive_weight)
    )


def decoded_spans(logits, valid, *, threshold, blocked=()):
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be in (0, 1)")
    margin = math.log(threshold / (1 - threshold))
    indices = (valid & (logits > margin)).nonzero().cpu().tolist()
    scores = logits.detach().cpu()
    return choose_spans(
        [
            SpanCandidate(start, start + width + 1, float(scores[start, width]))
            for start, width in indices
        ],
        margin=margin,
        blocked=blocked,
    )
