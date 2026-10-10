"""Real masking integration for the frozen-E5 generalization candidate."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from name_generalization_adapter import GeneralizationNER, build_ner
finally:
    sys.path[:] = previous_path


def adapter():
    import torch

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

    class Head:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, features, chars, positions, prior, lengths):
            assert features.dtype == torch.float16
            assert prior.shape == (*chars.shape, 5)
            logits = torch.full((*chars.shape, 5), -20.0)
            # Address also proposed as a name: it must remain an address.
            for i, char in enumerate(chars[0].tolist()):
                label = {1: 4, 3: 1, 4: 2, 5: 3, 6: 4}.get(char, 0)
                logits[0, i, label] = 20.0
            return logits

    class Model:
        config = SimpleNamespace(id2label={0: "O", 1: "S-private_address"})

        def __init__(self):
            self.calls = 0

        def __call__(self, **kwargs):
            assert kwargs["output_hidden_states"]
            self.calls += 1
            hidden = torch.zeros(kwargs["input_ids"].shape[0], 6, 8)
            logits = torch.zeros(hidden.shape[0], 6, 2)
            logits[:, :, 0] = 20
            logits[0, 1] = torch.tensor([0.0, 20.0])
            return SimpleNamespace(logits=logits, hidden_states=(hidden,))

    def forbidden_old_analyze(text):
        raise AssertionError("Adapter must share the one encoder forward")

    backbone = SimpleNamespace(
        device="cpu", tokenizer=Tokenizer(), max_length=6, stride=2, batch_size=2,
        model=Model(), analyze=forbidden_old_analyze, score_threshold=0.9,
        name_context_head=None,
    )
    return GeneralizationNER(backbone, Head(), dict(zip("가나다라마바", range(1, 7),
                                                      strict=True)), threshold=0.9)


def test_global_character_decode_crosses_window_boundary_and_preserves_addresses():
    result = adapter().analyze("가나다라마바")
    assert [(r.entity_type, r.start, r.end) for r in result] == [
        ("KR_ADDRESS", 0, 1), ("KR_NAME", 2, 5), ("KR_NAME", 5, 6),
    ]


def test_actual_public_stars_mask_preserves_original_coordinates():
    from ko_pii_guard import KoreanPIIGuard

    guard = KoreanPIIGuard(entities=["KR_NAME"], ner=adapter(), score_threshold=0.0)
    assert guard.mask("가나다라마바", style="stars") == "가나****"


def test_names_and_addresses_share_one_batched_backbone_forward():
    ner = adapter()
    ner.analyze("가나다라마바")
    assert ner.backbone.model.calls == 1


def test_empty_text_does_not_invoke_encoder():
    ner = adapter()
    assert ner.analyze(" \n ") == []
    assert ner.backbone.model.calls == 0


def test_factory_rejects_checkpoint_for_another_backbone_before_loading(tmp_path):
    from ko_pii_guard.ner import MODEL_ID

    (tmp_path / "config.json").write_text(json.dumps({
        "model_id": MODEL_ID, "revision": "wrong-revision", "threshold": 0.9,
    }))
    with pytest.raises(ValueError, match="pinned E5"):
        build_ner({"checkpoint": str(tmp_path)})
