"""Live whole-span policies must retain raw-E5 guards and isolated anchor state."""

import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from presidio_analyzer import RecognizerResult

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from name_adapted_adapter import AdaptedNER
    from name_encoder_adaptation import LoRALinear
    from name_multisource_adapter import MultiSourceNER, build_ner
finally:
    sys.path[:] = previous_path


def make_adapter(*, protect=True, correction=0., threshold=0.5, before_anchor=None):
    class Tokenizer:
        def __call__(self, text, **kwargs):
            offsets = [[0, 0], *[[i, i + 1] for i in range(len(text))], [0, 0]]
            return dict(input_ids=torch.zeros(1, len(offsets), dtype=torch.long),
                        attention_mask=torch.ones(1, len(offsets), dtype=torch.long),
                        offset_mapping=torch.tensor([offsets]))

    class Model(torch.nn.Module):
        config = SimpleNamespace(id2label={0: "O", 1: "S-private_person"})

        def __init__(self):
            super().__init__()
            self.adapter = LoRALinear(torch.nn.Linear(1, 1), rank=1, alpha=1)
            self.original_calls = self.adapted_calls = 0

        def forward(self, **kwargs):
            assert not self.adapter.enabled
            self.original_calls += 1
            logits = torch.full((*kwargs["input_ids"].shape, 2), -20.)
            logits[:, :, 0] = 20.
            return SimpleNamespace(logits=logits)

        def base_model(self, **kwargs):
            assert self.adapter.enabled
            self.adapted_calls += 1
            return SimpleNamespace(last_hidden_state=torch.zeros(
                *kwargs["input_ids"].shape, 8))

    class Head:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, features, chars, positions, prior, lengths):
            logits = torch.full((*chars.shape, 5), -20.)
            logits[:, :, 0] = 20.
            return logits

    model = Model()

    def original_names(text):
        assert not model.adapter.enabled
        if before_anchor is not None:
            before_anchor(text)
        model(input_ids=torch.zeros(1, len(text) + 2, dtype=torch.long))
        return [RecognizerResult("KR_NAME", index, index + 1, 0.98765)
                for index, char in enumerate(text) if char in "가2"]

    backbone = SimpleNamespace(device="cpu", model=model, tokenizer=Tokenizer(),
                               analyze=original_names, score_threshold=0.9,
                               name_context_head=None, batch_size=4, stride=2, max_length=32)
    inner = AdaptedNER(backbone, Head(), {"가": 2, "나": 3}, threshold=threshold,
                       encoder_autocast="float32")
    return MultiSourceNER(inner, threshold=threshold, class_logit_correction=correction,
                          protect_anchors=protect)


def test_original_lora_disabled_public_anchors_filter_non_names_before_protection():
    ner = make_adapter()
    findings = ner.analyze("가2나")
    assert [(r.start, r.end, r.score) for r in findings] == [(0, 1, 0.988)]
    assert ner.backbone.model.adapter.enabled
    assert ner.backbone.model.original_calls == 2
    assert ner.backbone.model.adapted_calls == 1


def test_actual_stars_mask_uses_protected_public_name_without_numeric_false_anchor():
    from ko_pii_guard import KoreanPIIGuard

    guard = KoreanPIIGuard(entities=["KR_NAME"], ner=make_adapter(), score_threshold=0.)
    assert guard.mask("가2나", style="stars") == "*2나"


def test_analytical_class_correction_does_not_mutate_original_logits():
    uncorrected = make_adapter(protect=False, correction=0., threshold=0.3)
    corrected = make_adapter(protect=False, correction=math.log(4), threshold=0.3)
    logits = torch.zeros(1, 5)
    # Legal one-character paths are only O and S: 1/2 becomes 1/(1+4).
    assert uncorrected.decode_logits(logits) == [(0, 1, 0.5)]
    assert corrected.decode_logits(logits) == []
    assert torch.equal(logits, torch.zeros(1, 5))


def test_anchor_context_and_lora_state_are_cleared_after_an_exception():
    ner = make_adapter()

    class BrokenHead:
        def __call__(self, *args):
            raise RuntimeError("head failure")

    ner.head = BrokenHead()
    with pytest.raises(RuntimeError, match="head failure"):
        ner.analyze("가 나")
    assert ner.backbone.model.adapter.enabled
    assert ner.decode_logits(torch.tensor([[20., -20., -20., -20., -20.]])) == []


def test_concurrent_anchor_requests_cannot_mix_per_document_coordinates():
    entered, attempted, second_entered, release = (threading.Event() for _ in range(4))

    def before_anchor(text):
        if text == "가 나":
            entered.set()
            assert release.wait(5)
        else:
            second_entered.set()

    ner = make_adapter(before_anchor=before_anchor)

    def second_request():
        attempted.set()
        return ner.analyze("나 가")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(ner.analyze, "가 나")
        try:
            assert entered.wait(5)
            second = executor.submit(second_request)
            assert attempted.wait(5)
            assert not second_entered.wait(0.1)
        finally:
            release.set()
        assert [(r.start, r.end) for r in first.result(timeout=5)] == [(0, 1)]
        assert [(r.start, r.end) for r in second.result(timeout=5)] == [(2, 3)]
    assert ner.backbone.model.adapter.enabled


def test_without_anchors_only_two_encoder_passes_and_empty_input_no_passes():
    ner = make_adapter(protect=False)
    assert ner.analyze("가2나") == []
    assert ner.backbone.model.original_calls == ner.backbone.model.adapted_calls == 1
    assert ner.analyze(" \n ") == []
    assert ner.backbone.model.original_calls == ner.backbone.model.adapted_calls == 1


def test_factory_requires_explicit_policy_before_loading_any_checkpoint():
    with pytest.raises(ValueError, match="Explicit"):
        build_ner({"checkpoint": "unused", "model_kind": "adapted", "threshold": 0.5})
