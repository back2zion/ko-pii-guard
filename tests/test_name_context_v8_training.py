"""Mixed person/title errors are mandatory negatives, not just sampled prose."""

import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_mixed_person_false_span_must_be_removed_without_rejecting_real_person():
    torch = pytest.importorskip("torch")
    previous_path = sys.path[:]
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        samples = runpy.run_path(str(ROOT / "scripts/train_name_context_v8.py"))[
            "candidate_samples"
        ]
    finally:
        sys.path[:] = previous_path

    class Head:
        def __call__(self, features, characters, positions, lengths):
            logits = torch.full((1, 5, 5), -20.)
            for i, label in enumerate([1, 3, 0, 1, 3]):
                logits[0, i, label] = 20.
            return logits

    cases = [dict(text="갑을 온기", expected=[dict(entity="KR_NAME", start=0, end=2)])]
    cache = [(torch.ones(5, 768), torch.ones(5, dtype=torch.long),
              torch.zeros(5, 2), torch.tensor([1, 3, 0, 0, 0]))]
    values, labels, required = samples(cases, cache, Head(), Head())
    assert values.shape == (2, 3842)
    assert labels.tolist() == [0, 1]
    assert required.tolist() == [False, True]
