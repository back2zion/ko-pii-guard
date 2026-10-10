"""Check constrained decoding against exhaustive legal paths, not its implementation."""

import itertools
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from name_bioes_ablation import constrained_bioes
finally:
    sys.path[:] = previous_path


def legal(path):
    open_name = False
    for label in path:
        if label in (0, 4):
            if open_name:
                return False
        elif label == 1:
            if open_name:
                return False
            open_name = True
        elif label == 2:
            if not open_name:
                return False
        elif label == 3:
            if not open_name:
                return False
            open_name = False
    return not open_name


@pytest.mark.parametrize("length", range(1, 6))
def test_viterbi_matches_exhaustive_legal_sequences(length):
    torch.manual_seed(length)
    logits = torch.randn(length, 5)
    result = constrained_bioes(logits).tolist()
    assert legal(result)
    best = max(
        sum(float(logits[i, tag]) for i, tag in enumerate(path))
        for path in itertools.product(range(5), repeat=length)
        if legal(path)
    )
    assert sum(float(logits[i, tag]) for i, tag in enumerate(result)) == pytest.approx(best)


def test_orphan_inside_tag_cannot_be_a_single_character_name():
    logits = torch.tensor([[0.0, -1.0, 9.0, 8.0, 2.0]])
    assert constrained_bioes(logits).tolist() == [4]


def test_empty_input_and_invalid_logit_shape():
    assert constrained_bioes(torch.empty(0, 5)).tolist() == []
    with pytest.raises(ValueError):
        constrained_bioes(torch.zeros(2, 4))
