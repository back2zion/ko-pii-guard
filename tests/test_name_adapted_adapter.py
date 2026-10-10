"""Adapted features must never change the original E5 prior or addresses."""

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import name_adapted_adapter
    from name_adapted_adapter import AdaptedNER, build_frozen_ner, build_ner
    from name_encoder_adaptation import LoRALinear
finally:
    sys.path[:] = previous_path


def adapter(*, decoder="viterbi", on_logits=None):
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert kwargs["return_overflowing_tokens"]
            return dict(
                input_ids=torch.zeros(2, 6, dtype=torch.long),
                attention_mask=torch.ones(2, 6, dtype=torch.long),
                offset_mapping=torch.tensor([
                    [[0, 0], [0, 1], [1, 2], [2, 3], [3, 4], [0, 0]],
                    [[0, 0], [2, 3], [3, 4], [4, 5], [5, 6], [0, 0]],
                ]),
            )

    class Model(torch.nn.Module):
        config = SimpleNamespace(id2label={0: "O", 1: "S-private_address", 2: "S-private_person"})

        def __init__(self):
            super().__init__()
            self.adaptation = LoRALinear(torch.nn.Linear(1, 1, bias=False), rank=1, alpha=1)
            with torch.no_grad():
                self.adaptation.base.weight.fill_(1.)
                self.adaptation.lora_A.fill_(1.)
                self.adaptation.lora_B.fill_(2.)
            self.base_calls = self.adapted_calls = 0

        def forward(self, **kwargs):
            self.base_calls += 1
            # A non-disabled pass would corrupt the frozen person prior.
            logits = torch.full((*kwargs["input_ids"].shape, 3), -20.)
            logits[:, :, 2 if self.adaptation.enabled else 0] = 20.
            logits[0, 1] = torch.tensor([-20., 20., -20.])
            return SimpleNamespace(logits=logits)

        def base_model(self, **kwargs):
            self.adapted_calls += 1
            inputs = torch.ones(*kwargs["input_ids"].shape, 1)
            return SimpleNamespace(last_hidden_state=self.adaptation(inputs).expand(-1, -1, 8))

    class Head:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, features, chars, positions, prior, lengths):
            assert features.dtype == torch.float16
            assert torch.all(features == 3.)  # Adaptation is active only for hidden states.
            assert torch.all(prior[:, :, 0] > 0.999)  # Original E5 has no person proposals.
            logits = torch.full((*chars.shape, 5), -20.)
            for i, char in enumerate(chars[0].tolist()):
                logits[0, i, {1: 4, 3: 1, 4: 2, 5: 3, 6: 4}.get(char, 0)] = 20.
            return logits

    backbone = SimpleNamespace(
        device="cpu", tokenizer=Tokenizer(), model=Model(), max_length=6,
        stride=2, batch_size=2, score_threshold=0.9, name_context_head=None,
    )
    return AdaptedNER(backbone, Head(), dict(zip("가나다라마바", range(1, 7), strict=True)),
                      threshold=0.5, decoder=decoder, on_logits=on_logits)


def test_two_passes_keep_original_prior_and_addresses_with_adapted_hidden_features():
    ner = adapter()
    findings = ner.analyze("가나다라마바")
    assert [(r.entity_type, r.start, r.end) for r in findings] == [
        ("KR_ADDRESS", 0, 1), ("KR_NAME", 2, 5), ("KR_NAME", 5, 6),
    ]
    assert ner.backbone.model.base_calls == ner.backbone.model.adapted_calls == 1
    assert ner.backbone.model.adaptation.enabled


def test_actual_stars_mask_uses_original_coordinates_across_windows():
    from ko_pii_guard import KoreanPIIGuard

    guard = KoreanPIIGuard(entities=["KR_NAME"], ner=adapter(), score_threshold=0.)
    assert guard.mask("가나다라마바", style="stars") == "가나****"


def test_coverage_decoder_is_explicit_and_logits_callback_observes_merged_document():
    captured = []
    ner = adapter(decoder="coverage", on_logits=lambda logits: captured.append(logits.clone()))
    findings = ner.analyze("가나다라마바")
    assert [(r.entity_type, r.start, r.end) for r in findings] == [
        ("KR_ADDRESS", 0, 1), ("KR_NAME", 2, 6),
    ]
    assert len(captured) == 1
    assert captured[0].device.type == "cpu"
    assert captured[0].shape == (6, 5)
    assert ner.decoder_name == "coverage"


def test_empty_text_does_not_invoke_either_encoder_pass():
    ner = adapter()
    assert ner.analyze(" \n ") == []
    assert ner.backbone.model.base_calls == ner.backbone.model.adapted_calls == 0


def test_unknown_decoder_rejected_before_inference():
    with pytest.raises(ValueError, match="decoder"):
        adapter(decoder="unknown")


def test_factory_rejects_wrong_backbone_before_weights_loading(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_id": "wrong", "revision": "wrong", "threshold": 0.5,
    }))
    with pytest.raises(ValueError, match="pinned E5"):
        build_ner(dict(checkpoint=str(tmp_path)))


def test_logits_observer_cannot_change_predictions_by_mutating_its_tensor():
    ner = adapter(on_logits=lambda logits: logits.zero_())
    findings = ner.analyze("가나다라마바")
    assert [(r.start, r.end) for r in findings if r.entity_type == "KR_NAME"] == [(2, 5), (5, 6)]


def test_single_window_keeps_leading_and_trailing_space_context_from_training():
    ner = adapter()
    ner.vocabulary = {"가": 2}
    ner.backbone.tokenizer = lambda text, **kwargs: dict(
        input_ids=torch.zeros(1, 3, dtype=torch.long),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        offset_mapping=torch.tensor([[[0, 0], [1, 2], [0, 0]]]),
    )

    def original_outside(**kwargs):
        assert not ner.backbone.model.adaptation.enabled
        logits = torch.full((1, 3, 3), -20.)
        logits[:, :, 0] = 20.
        return SimpleNamespace(logits=logits)

    ner.backbone.model.forward = original_outside

    class SpaceHead:
        def __call__(self, features, chars, positions, prior, lengths):
            assert lengths.tolist() == [3]
            assert chars.tolist() == [[1, 2, 1]]
            assert features[0, :, 0].tolist() == [0., 3., 0.]
            assert torch.all(prior[:, :, 0] > 0.999)
            logits = torch.full((1, 3, 5), -20.)
            logits[0, range(3), [0, 4, 0]] = 20.
            return logits

    ner.head = SpaceHead()
    from ko_pii_guard import KoreanPIIGuard

    guard = KoreanPIIGuard(entities=["KR_NAME"], ner=ner, score_threshold=0.)
    assert [(r.start, r.end) for r in guard.analyze(" 가 ")] == [(1, 2)]
    assert guard.mask(" 가 ", style="stars") == " * "


def test_concurrent_requests_do_not_interleave_temporary_lora_state():
    ner = adapter()
    first_entered, second_entered = threading.Event(), threading.Event()
    second_attempted, release = threading.Event(), threading.Event()
    counter_lock, counter = threading.Lock(), [0]
    original_forward = ner.backbone.model.forward

    def blocked_forward(**kwargs):
        with counter_lock:
            counter[0] += 1
            number = counter[0]
        if number == 1:
            first_entered.set()
            if not release.wait(5):
                raise RuntimeError("First request was not released")
        else:
            second_entered.set()
        return original_forward(**kwargs)

    ner.backbone.model.forward = blocked_forward

    def second_request():
        second_attempted.set()
        return ner.analyze("가나다라마바")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(ner.analyze, "가나다라마바")
        try:
            assert first_entered.wait(5)
            second = executor.submit(second_request)
            assert second_attempted.wait(5)
            assert not second_entered.wait(0.1)
        finally:
            release.set()
        for result in (first.result(timeout=5), second.result(timeout=5)):
            assert [(r.entity_type, r.start, r.end) for r in result] == [
                ("KR_ADDRESS", 0, 1), ("KR_NAME", 2, 5), ("KR_NAME", 5, 6),
            ]
    assert ner.backbone.model.adaptation.enabled
    assert ner.backbone.model.base_calls == ner.backbone.model.adapted_calls == 2


def test_frozen_factory_loads_original_head_without_installing_lora(tmp_path, monkeypatch):
    from ko_pii_guard.ner import MODEL_ID, MODEL_REVISION

    checkpoint = dict(model_id=MODEL_ID, revision=MODEL_REVISION, vocabulary={"가": 2},
                      hidden_size=8, projection_size=4, char_embedding_size=3,
                      lstm_hidden_size=4, threshold=0.9)
    (tmp_path / "config.json").write_text(json.dumps(checkpoint))
    observed = {}

    class Head:
        def __init__(self, **kwargs):
            observed["architecture"] = kwargs

        def load_state_dict(self, state):
            observed["state"] = state

        def to(self, device):
            return self

        def eval(self):
            return self

    backbone = SimpleNamespace(device="cpu", name_context_head=None)
    monkeypatch.setattr(name_adapted_adapter, "GeneralNameHead", Head)
    monkeypatch.setattr(name_adapted_adapter, "load_file", lambda path: {"frozen": path})
    monkeypatch.setattr(name_adapted_adapter.KoreanNER, "from_pretrained",
                        lambda **kwargs: backbone)

    def forbidden_injection(*args, **kwargs):
        raise AssertionError("Frozen fallback must never install LoRA")

    monkeypatch.setattr(name_adapted_adapter, "inject_lora", forbidden_injection)
    ner = build_frozen_ner(dict(checkpoint=str(tmp_path), threshold=0.1))
    assert ner.backbone is backbone
    assert ner.threshold == 0.1
    assert ner.decoder_name == "coverage"
    assert ner.encoder_autocast == "float32"
    assert observed["architecture"]["char_vocab_size"] == 3
    assert observed["state"] == {"frozen": str(tmp_path / "head.safetensors")}
    with pytest.raises(ValueError, match="explicit threshold"):
        build_frozen_ner(dict(checkpoint=str(tmp_path)))


def test_frozen_no_lora_backbone_preserves_original_features_and_prior():
    ner = adapter()
    # Replace the installed adapter by its original linear layer. This is a
    # separate, unadapted backbone; the generic two-pass path must still work.
    ner.backbone.model.adaptation = ner.backbone.model.adaptation.base

    def original_outside(**kwargs):
        logits = torch.full((*kwargs["input_ids"].shape, 3), -20.)
        logits[:, :, 0] = 20.
        return SimpleNamespace(logits=logits)

    ner.backbone.model.forward = original_outside
    ner.encoder_autocast = "float32"

    class FrozenHead:
        def __call__(self, features, chars, positions, prior, lengths):
            assert torch.all(features == 1.)
            assert torch.all(prior[:, :, 0] > 0.999)
            logits = torch.full((*chars.shape, 5), -20.)
            logits[:, :, 0] = 20.
            return logits

    ner.head = FrozenHead()
    assert ner.analyze("가나다라마바") == []
    assert ner.backbone.model.adapted_calls == 1
