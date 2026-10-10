"""Research character decoder with an explicit pretrained person-label prior.

Only PyTorch is required; this module does not import the public masking runtime.
Character labels are O=0, B=1, I=2, E=3, S=4 throughout.
"""

from __future__ import annotations

import math

import torch
from name_bioes_ablation import constrained_bioes
from torch import nn


def _offset_owners(offsets, left, right):
    """Assign each character to the overlapping token with most context.

    Zero-width special tokens are ignored. Ties retain the earlier token. The
    same ownership is used for hidden features and the pretrained label prior.
    Clipping a window never changes a token's original character boundaries.
    """
    if not isinstance(left, int) or not isinstance(right, int) or not 0 <= left <= right:
        raise ValueError("Expected a nonnegative, ordered character window")
    pairs = offsets.tolist() if isinstance(offsets, torch.Tensor) else list(offsets)
    valid = []
    for index, pair in enumerate(pairs):
        if len(pair) != 2:
            raise ValueError("Each token offset must contain two coordinates")
        start, end = pair
        if (not isinstance(start, int) or not isinstance(end, int)
                or not 0 <= start <= end):
            raise ValueError("Token offsets must be nonnegative and ordered")
        if end > start:
            valid.append(index)
    owners, ranks = [-1] * (right - left), [-1] * (right - left)
    for rank_index, index in enumerate(valid):
        start, end = pairs[index]
        rank = min(rank_index, len(valid) - rank_index - 1)
        for position in range(max(start, left), min(end, right)):
            local = position - left
            if rank > ranks[local]:
                owners[local], ranks[local] = index, rank
    return pairs, owners


def encode_character_features(text, hidden, offsets, left=0, right=None):
    """Expand token features at original offsets; uncovered characters are zero.

    Relative positions use the original token width, including for partial
    windows. No normalization, interpolation, or lexical rules are applied.
    """
    right = len(text) if right is None else right
    pairs, owners = _offset_owners(offsets, left, right)
    if (hidden.ndim != 2 or len(hidden) != len(pairs) or right > len(text)
            or any(end > len(text) for _, end in pairs)):
        raise ValueError("Hidden states and offsets must match the original text")
    features = hidden.new_zeros((right - left, hidden.shape[-1]))
    positions = hidden.new_zeros((right - left, 2))
    covered = [i for i, owner in enumerate(owners) if owner >= 0]
    if covered:
        indices = torch.tensor(covered, dtype=torch.long, device=hidden.device)
        token_indices = torch.tensor([owners[i] for i in covered], device=hidden.device)
        features[indices] = hidden[token_indices]
        positions[indices] = hidden.new_tensor([
            ((i + left - pairs[owners[i]][0]) / (pairs[owners[i]][1] - pairs[owners[i]][0]),
             (pairs[owners[i]][1] - i - left - 1) / (pairs[owners[i]][1] - pairs[owners[i]][0]))
            for i in covered
        ])
    return features, positions


def character_prior(token_probs, offsets, left, right, id2label):
    """Map every upstream class probability to character BIOES probabilities.

    A multi-character S token becomes B/I/E; token B and E contribute their
    boundary mass only to the first and last character respectively. Other
    entity classes collapse to O. Uncovered characters, including spaces
    between tokens, have probability one for O.
    """
    pairs, owners = _offset_owners(offsets, left, right)
    labels = {int(key): value for key, value in id2label.items()}
    if (token_probs.ndim != 2 or len(token_probs) != len(pairs)
            or set(labels) != set(range(token_probs.shape[-1]))
            or not token_probs.is_floating_point() or not torch.isfinite(token_probs).all()
            or (token_probs < 0).any() or (token_probs > 1).any()
            or not torch.allclose(token_probs.sum(-1), token_probs.new_ones(len(pairs)),
                                  atol=0.002, rtol=0.002)):
        raise ValueError("Expected finite, normalized token probabilities and matching labels")
    prior = token_probs.new_zeros((right - left, 5))
    prior[:, 0] = 1
    covered = [i for i, owner in enumerate(owners) if owner >= 0]
    if not covered:
        return prior
    indices = torch.tensor(covered, dtype=torch.long, device=token_probs.device)
    token_indices = torch.tensor([owners[i] for i in covered], device=token_probs.device)
    first = torch.tensor([i + left == pairs[owners[i]][0] for i in covered],
                         dtype=torch.bool, device=token_probs.device)
    last = torch.tensor([i + left == pairs[owners[i]][1] - 1 for i in covered],
                        dtype=torch.bool, device=token_probs.device)
    expanded = token_probs.new_zeros((len(covered), 5))
    for label_index, label in labels.items():
        prefix, _, entity = label.partition("-")
        target = torch.zeros(len(covered), dtype=torch.long, device=token_probs.device)
        if entity in ("private_person", "NAME"):
            if prefix == "B":
                target[:] = 2
                target[first] = 1
            elif prefix == "I":
                target[:] = 2
            elif prefix == "E":
                target[:] = 2
                target[last] = 3
            elif prefix == "S":
                target[:] = 2
                target[first] = 1
                target[last] = 3
                target[first & last] = 4
        expanded.scatter_add_(1, target[:, None], token_probs[token_indices, label_index, None])
    prior[indices] = expanded
    return prior


class GeneralNameHead(nn.Module):
    """Character residual classifier over frozen contextual features and priors."""

    def __init__(self, *, char_vocab_size, hidden_size=768, projection_size=64,
                 char_embedding_size=32, lstm_hidden_size=64):
        super().__init__()
        if min(char_vocab_size, hidden_size, projection_size, char_embedding_size,
               lstm_hidden_size) < 1:
            raise ValueError("Head dimensions must be positive")
        self.projection = nn.Linear(hidden_size, projection_size)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_size, padding_idx=0)
        self.recurrent = nn.LSTM(projection_size + char_embedding_size + 2 + 5,
                                 lstm_hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(2 * lstm_hidden_size, 5)
        self.prior_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, features, chars, positions, prior, lengths):
        if (features.ndim != 3 or chars.shape != features.shape[:2]
                or positions.shape != (*features.shape[:2], 2)
                or prior.shape != (*features.shape[:2], 5)
                or lengths.ndim != 1 or len(lengths) != len(features)
                or (lengths < 1).any() or (lengths > features.shape[1]).any()):
            raise ValueError("Expected matching padded character features and positive lengths")
        dtype = self.projection.weight.dtype
        features, positions, prior = (value.to(dtype=dtype)
                                      for value in (features, positions, prior))
        inputs = torch.cat((self.projection(features), self.char_embedding(chars.long()),
                            positions, prior), dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(inputs, lengths.detach().cpu(),
                                                 batch_first=True, enforce_sorted=False)
        encoded, _ = self.recurrent(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True,
                                                    total_length=features.shape[1])
        return self.classifier(encoded) + self.prior_scale * prior.clamp_min(1e-5).log()


def decode_char_logits(logits, threshold=0.9, origin=0):
    """Decode a legal BIOES sequence into original spans and mean confidence."""
    if (not math.isfinite(threshold) or not 0 <= threshold <= 1
            or not isinstance(origin, int) or origin < 0):
        raise ValueError("Expected a probability threshold and nonnegative original offset")
    path = constrained_bioes(logits)
    probabilities = logits.softmax(-1)
    scores = probabilities.gather(1, path[:, None]).flatten().tolist()
    labels, spans, start = path.tolist(), [], None
    for index, label in enumerate(labels):
        if label == 1:
            start = index
        elif label == 3:
            score = sum(scores[start:index + 1]) / (index + 1 - start)
            if score >= threshold:
                spans.append((origin + start, origin + index + 1, score))
            start = None
        elif label == 4 and scores[index] >= threshold:
            spans.append((origin + index, origin + index + 1, scores[index]))
    return spans
