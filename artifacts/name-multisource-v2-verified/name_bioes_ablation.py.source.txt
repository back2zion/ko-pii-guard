"""Research ablation: legal BIOES transitions on frozen character logits.

A structurally valid path does not imply correct entity boundaries or semantics.
"""

from __future__ import annotations

import torch

# O=0, B=1, I=2, E=3, S=4; rows are previous tags, columns next tags.
TRANSITIONS = ((0, 1, 4), (2, 3), (2, 3), (0, 1, 4), (0, 1, 4))


def constrained_bioes(logits):
    """Return the highest-scoring legal label sequence, with closed entity ends."""
    if logits.ndim != 2 or logits.shape[1] != 5 or not torch.isfinite(logits).all():
        raise ValueError("Expected finite character logits with five BIOES classes")
    if not len(logits):
        return torch.empty(0, dtype=torch.long, device=logits.device)
    transition = logits.new_full((5, 5), -torch.inf)
    for previous, following in enumerate(TRANSITIONS):
        transition[previous, list(following)] = 0
    state = logits[0].clone()
    state[[2, 3]] = -torch.inf
    history = []
    for row in logits[1:]:
        score, parent = (state[:, None] + transition).max(0)
        state = score + row
        history.append(parent)
    state[[1, 2]] = -torch.inf
    final = state.argmax()
    path = [final]
    for parent in reversed(history):
        final = parent[final]
        path.append(final)
    return torch.stack(list(reversed(path)))
