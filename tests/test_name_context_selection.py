"""Selection rejects recall regressions even when aggregate accuracy is unchanged."""

import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def api():
    pytest.importorskip("torch")
    previous = sys.path[:]
    try:
        return runpy.run_path(str(ROOT / "scripts/train_name_context.py"))
    finally:
        sys.path[:] = previous


def test_same_tp_fn_totals_cannot_hide_a_lost_previous_name(api):
    import torch

    class Head:
        def __init__(self, labels):
            self.labels = labels

        def eval(self):
            return self

        def __call__(self, features, chars, positions, lengths):
            logits = torch.full((3, 1, 5), -20.0)
            for index, label in enumerate(self.labels):
                logits[index, 0, label] = 20.0
            return logits

    cases = [dict(id=str(i), text=text, expected=[dict(entity="KR_NAME", start=0, end=1)]
                  if i < 2 else []) for i, text in enumerate(("갑", "을", "말"))]
    cached = [(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long),
               torch.zeros(1, 2), torch.zeros(1, dtype=torch.long)) for _ in cases]
    baseline = api["score_head"](Head([4, 0, 0]), cached, cases, .9, collect_spans=True)
    protected = {tuple(span) for span in baseline["matched_gold"]}
    candidate = api["score_head"](Head([0, 4, 0]), cached, cases, .9, protected=protected)
    assert (candidate["tp"], candidate["fn"]) == (baseline["tp"], baseline["fn"]) == (1, 1)
    assert candidate["protected_gold_lost"] == 1
    assert not api["checkpoint_is_eligible"](candidate, dict(fp=0, fn=0), baseline, True)


@pytest.mark.parametrize("change", [dict(fp=2), dict(protected_gold_lost=1),
                                   dict(negative_false_positive_sentences=1)])
def test_no_metric_improvement_can_override_a_regression_gate(api, change):
    baseline = dict(fp=1, negative_false_positive_sentences=0)
    metrics = dict(fp=0, protected_gold_lost=0, negative_false_positive_sentences=0, f1=1)
    assert api["checkpoint_is_eligible"](metrics, dict(fp=0, fn=0), baseline, True)
    assert not api["checkpoint_is_eligible"](metrics | change, dict(fp=0, fn=0), baseline, True)
    assert not api["checkpoint_is_eligible"](metrics, dict(fp=0, fn=1), baseline, True)
