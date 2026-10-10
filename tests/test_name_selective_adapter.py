"""Selective changes must use the shared public anchor path and explicit policy."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from presidio_analyzer import RecognizerResult

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import name_selective_adapter as adapter
    from name_adapted_adapter import AdaptedNER
finally:
    sys.path[:] = previous_path


def make_inner():
    backbone = SimpleNamespace(
        device="cpu", score_threshold=.9, model=torch.nn.Linear(1, 1),
        analyze=lambda text: [RecognizerResult("KR_NAME", 0, 1, .98765)],
    )
    return AdaptedNER(backbone, torch.nn.Identity(), {}, threshold=.9,
                      encoder_autocast="float32")


def test_selective_factory_rejects_implicit_or_weighted_policy_before_loading(monkeypatch):
    def forbidden(config):
        raise AssertionError("No checkpoint access for invalid policy")
    monkeypatch.setattr(adapter, "build_frozen_ner", forbidden)
    for config in (
        {},
        dict(addition_threshold=.99, removal_threshold=.99),
        dict(model_kind="adapted", addition_threshold=.99, removal_threshold=.99,
             class_logit_correction=0.),
        dict(model_kind="frozen", addition_threshold=.99, removal_threshold=True,
             class_logit_correction=0.),
        dict(model_kind="frozen", addition_threshold=.99, removal_threshold=.99,
             class_logit_correction=1.),
    ):
        with pytest.raises(ValueError):
            adapter.build_ner(config)


def test_factory_keeps_shared_lock_and_unweighted_logits_observer(monkeypatch):
    inner = make_inner()
    observed = []
    inner.on_logits = lambda values: (observed.append(values.clone()), values.fill_(99))
    monkeypatch.setattr(adapter, "build_frozen_ner", lambda config: inner)
    ner = adapter.build_ner(dict(model_kind="frozen", addition_threshold=1.,
                                  removal_threshold=1., class_logit_correction=0.))
    assert ner._inference_lock is inner._inference_lock
    ner._anchors = ((0, 1, .7),)
    logits = torch.tensor([[20., -20., -20., -20., -20.]])
    original = logits.clone()
    assert ner.decode_logits(logits) == [(0, 1, .7)]
    assert torch.equal(logits, original)
    assert torch.equal(observed[0], original)


def test_actual_mask_selectively_removes_anchor_and_restores_request_state(monkeypatch):
    from ko_pii_guard import KoreanPIIGuard

    ner = adapter.SelectiveNER(make_inner(), addition_threshold=.99, removal_threshold=.99)
    logits = torch.tensor([[20., -20., -20., -20., -20.],
                           [-20., -20., -20., -20., 20.]])

    def infer(self, text):
        return [RecognizerResult("KR_NAME", start, end, score)
                for start, end, score in ner.decode_logits(logits)]
    monkeypatch.setattr(AdaptedNER, "_analyze_locked", infer)
    guard = KoreanPIIGuard(entities=["KR_NAME"], score_threshold=0., ner=ner)
    assert guard.mask("가나", style="stars") == "가*"
    assert ner._anchors == ()
    assert ner._retained_anchors == ()


def test_all_o_evidence_below_removal_threshold_preserves_anchor():
    ner = adapter.SelectiveNER(make_inner(), addition_threshold=1., removal_threshold=.99)
    ner._anchors = ((0, 1, .7),)
    assert ner.decode_logits(torch.zeros(1, 5)) == [(0, 1, .7)]


def test_retained_public_anchor_survives_address_veto_and_exception_clears_state(monkeypatch):
    from ko_pii_guard import KoreanPIIGuard

    ner = adapter.SelectiveNER(make_inner(), addition_threshold=1., removal_threshold=1.)

    def address_filtered(self, text):
        ner.decode_logits(torch.zeros(len(text), 5))
        return [RecognizerResult("KR_ADDRESS", 0, len(text), .99)]
    monkeypatch.setattr(AdaptedNER, "_analyze_locked", address_filtered)
    guard = KoreanPIIGuard(entities=["KR_NAME"], score_threshold=0., ner=ner)
    assert guard.mask("가나", style="stars") == "*나"
    assert ner._retained_anchors == ()

    def broken(self, text):
        ner.decode_logits(torch.zeros(len(text), 5))
        raise RuntimeError("failed after decoding")
    monkeypatch.setattr(AdaptedNER, "_analyze_locked", broken)
    with pytest.raises(RuntimeError, match="failed after decoding"):
        ner.analyze("가나")
    assert ner._anchors == ner._retained_anchors == ()
