"""Gold boundaries, padding, and validation selection use the intended units."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from train_name_generalization import collate, evaluate, gold_labels
finally:
    sys.path[:] = previous_path


def test_gold_supports_adjacent_names_singletons_and_token_internal_endpoints():
    case = dict(text="홍김가람이", expected=[
        dict(entity="KR_NAME", start=0, end=1),
        dict(entity="KR_NAME", start=1, end=4),
        dict(entity="PHONE_NUMBER", start=0, end=5),
    ])
    assert gold_labels(case).tolist() == [4, 1, 2, 3, 0]


@pytest.mark.parametrize("spans", [[(0, 2), (1, 3)], [(-1, 2)], [(0, 4)], [(2, 2)]])
def test_invalid_training_spans_fail_before_the_loss(spans):
    case = dict(text="가나다", expected=[dict(entity="KR_NAME", start=s, end=e)
                                          for s, e in spans])
    with pytest.raises(ValueError, match="gold name"):
        gold_labels(case)


def row(labels):
    length = len(labels)
    return (torch.zeros(length, 8).half(), torch.tensor(labels),
            torch.zeros(length, 2), torch.full((length, 5), 0.2), torch.tensor(labels))


def test_padding_contributes_neither_loss_nor_gradient():
    *_, lengths, target = collate([row([1, 3, 0]), row([4])], "cpu")
    assert lengths.tolist() == [3, 1]
    assert target.tolist() == [[1, 3, 0], [4, -100, -100]]
    logits = torch.randn(2, 3, 5, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten())
    loss.backward()
    assert torch.count_nonzero(logits.grad[1, 1:]) == 0
    assert torch.count_nonzero(logits.grad[1, 0]) > 0


def test_validation_ignores_padding_and_applies_span_confidence_after_legal_decode():
    class Head:
        def eval(self):
            return self

        def __call__(self, features, chars, positions, prior, lengths):
            probabilities = torch.full((*chars.shape, 5), 0.05)
            probabilities.scatter_(-1, chars[:, :, None], 0.8)
            # Unused padding deliberately looks like a high-confidence name.
            probabilities[1, 1:] = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.96])
            return probabilities.log()

    cases = [dict(text="가나다", expected=[dict(entity="KR_NAME", start=0, end=2)]),
             dict(text="홍", expected=[dict(entity="KR_NAME", start=0, end=1)])]
    scores = evaluate(Head(), [row([1, 3, 0]), row([4])], cases,
                      device="cpu", thresholds=[0.5, 0.9])
    assert scores["0.5"] == dict(tp=2, fp=0, fn=0, fully_covered_names=2, f1=1.0)
    assert scores["0.9"] == dict(tp=0, fp=0, fn=2, fully_covered_names=0, f1=0.0)
