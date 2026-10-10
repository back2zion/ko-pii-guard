"""Research windows must retain original offsets and exercise actual stars masking."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from presidio_analyzer import RecognizerResult

pytest.importorskip("torch")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from name_span_adapter import JointSpanNER
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
                offset_mapping=torch.tensor(
                    [
                        [[0, 0], [0, 1], [1, 2], [2, 3], [3, 4], [0, 0]],
                        [[0, 0], [2, 3], [3, 4], [4, 5], [5, 6], [0, 0]],
                    ]
                ),
            )

    class Head:
        widths = SimpleNamespace(num_embeddings=3)

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, features, chars, positions, lengths, *, max_span_width):
            n = chars.shape[1]
            logits = torch.full((1, n, max_span_width), -20.0)
            valid = torch.arange(n)[:, None] + torch.arange(max_span_width)[None, :] < n
            for i, char in enumerate(chars[0].tolist()):
                if char == 3 and i + 1 < n and chars[0, i + 1] == 4:
                    logits[0, i, 1] = 7.0
                if char == 6:
                    logits[0, i, 0] = 6.0
            return logits, valid[None]

    class Model:
        config = SimpleNamespace(id2label={0: "O", 1: "S-private_address"})

        def __init__(self):
            self.calls = 0

        def base_model(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                last_hidden_state=torch.zeros(kwargs["input_ids"].shape[0], 6, 8)
            )

        def __call__(self, **kwargs):
            hidden = self.base_model(**kwargs).last_hidden_state
            logits = torch.zeros(hidden.shape[0], 6, 2)
            logits[:, :, 0] = 20
            logits[0, 1] = torch.tensor([0.0, 20.0])
            return SimpleNamespace(logits=logits, hidden_states=(hidden,))

    model = Model()

    def old_analyze(text):
        model.calls += 1
        return [RecognizerResult("KR_ADDRESS", 0, 1, 0.95)]

    backbone = SimpleNamespace(
        device="cpu",
        tokenizer=Tokenizer(),
        max_length=6,
        stride=2,
        batch_size=2,
        model=model,
        analyze=old_analyze,
        score_threshold=0.9,
    )
    return JointSpanNER(
        backbone,
        Head(),
        dict(zip("가나다라마바", range(1, 7), strict=True)),
        max_span_width=2,
        threshold=0.5,
    )


def test_second_window_name_and_one_character_name_keep_original_offsets():
    result = adapter().analyze("가나다라마바")
    assert [(r.entity_type, r.start, r.end) for r in result] == [
        ("KR_ADDRESS", 0, 1),
        ("KR_NAME", 2, 4),
        ("KR_NAME", 5, 6),
    ]


def test_public_guard_stars_mask_exercises_research_adapter():
    from ko_pii_guard import KoreanPIIGuard

    guard = KoreanPIIGuard(entities=["KR_NAME"], ner=adapter(), score_threshold=0.0)
    assert guard.mask("가나다라마바", style="stars") == "가나**마*"


def test_empty_input_needs_no_model_calls():
    assert adapter().analyze(" \n ") == []


def test_addresses_and_joint_names_share_one_encoder_forward():
    ner = adapter()
    ner.analyze("가나다라마바")
    assert ner.backbone.model.calls == 1
