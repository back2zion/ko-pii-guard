"""Context model tests: no downloads, optional tensor and real-model checks."""

import os
from types import SimpleNamespace

import pytest

from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard
from ko_pii_guard.ner import KoreanNER, _decode_tokens


def test_bio_spans_keep_name_letters_and_split_adjacent_people():
    tokens = {
        (0, 2): (0, "B-NAME", .95),
        (2, 3): (1, "I-NAME", .95),  # 은 in 오지은 is part of the name
        (3, 4): (2, "O", .99),
        (5, 9): (3, "B-NAME", .9),
        (10, 13): (4, "B-NAME", .9),
        (14, 18): (5, "B-ADDR", .7),
    }
    results = _decode_tokens(tokens, .8)
    assert [(r.entity_type, r.start, r.end) for r in results] == [
        ("KR_NAME", 0, 3), ("KR_NAME", 5, 9), ("KR_NAME", 10, 13),
    ]


def test_other_pii_labels_do_not_join_names_or_become_names():
    tokens = {(0, 3): (0, "B-NAME", .95), (4, 8): (0, "B-SCHOOL", .99),
              (9, 12): (0, "I-NAME", .95)}
    assert [(r.start, r.end) for r in _decode_tokens(tokens, .8)] == [(0, 3), (9, 12)]


def test_bioes_respects_end_and_single_tags_without_stripping_name_letters():
    tokens = {(0, 1): (0, "B-private_person", .99),
              (1, 3): (1, "E-private_person", .99),
              (3, 4): (2, "O", .99),
              (5, 7): (3, "S-private_person", .99),
              (8, 12): (4, "S-private_address", .99)}
    assert [(r.entity_type, r.start, r.end) for r in _decode_tokens(tokens, .8)] == [
        ("KR_NAME", 0, 3), ("KR_NAME", 5, 7), ("KR_ADDRESS", 8, 12),
    ]


def test_overflow_keeps_later_names_and_prefers_window_interior():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        is_fast = True
        model_max_length = 6

        def __call__(self, text, **kwargs):
            assert kwargs["return_overflowing_tokens"] is True
            assert kwargs["stride"] == 2
            # Original offsets, overlapping characters 2 and 3; no truncation
            # of the name at character 5 in the second model window.
            return {
                "input_ids": torch.tensor([[0, 1, 2, 3, 4, 0], [0, 3, 4, 5, 6, 0]]),
                "attention_mask": torch.ones((2, 6), dtype=torch.long),
                "offset_mapping": torch.tensor([
                    [[0, 0], [0, 1], [1, 2], [2, 3], [3, 4], [0, 0]],
                    [[0, 0], [2, 3], [3, 4], [4, 5], [5, 6], [0, 0]],
                ]),
                "overflow_to_sample_mapping": torch.tensor([0, 0]),
            }

    class Model:
        config = SimpleNamespace(max_position_embeddings=6, id2label={0: "O", 1: "B-NAME"})

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, input_ids, attention_mask):
            assert not torch.is_grad_enabled()
            logits = torch.zeros((*input_ids.shape, 2))
            logits[:, :, 0] = 10
            # A window-edge false name at char2 must lose to the first
            # window's interior O. A real name beyond window1 must survive.
            for row in range(len(input_ids)):
                for col in range(len(input_ids[row])):
                    if input_ids[row, col] == 6 or (col == 1 and input_ids[row, col] == 3):
                        logits[row, col] = torch.tensor([0., 10.])
            return SimpleNamespace(logits=logits)

    ner = KoreanNER(Tokenizer(), Model(), stride=2, batch_size=1)
    assert [(r.start, r.end) for r in ner.analyze("가나다라마김")] == [(5, 6)]
    assert ner.analyze(" \n") == []


@pytest.mark.skipif(os.environ.get("KO_PII_TEST_NER") != "1",
                    reason="requires explicitly downloaded ner extra/model")
def test_real_model_distinguishes_homonyms_in_context():
    import torch

    torch.set_num_threads(2)
    guard = KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=KoreanNER.from_pretrained())
    assert not guard.contains_pii("이상 없습니다.")
    assert not guard.contains_pii("새해 소망을 적어 주세요.")
    assert guard.mask("이상 씨가 서류를 제출했습니다.") == "<KR_NAME> 씨가 서류를 제출했습니다."
    assert guard.mask("소망 씨가 내일 방문합니다.") == "<KR_NAME> 씨가 내일 방문합니다."
    assert guard.mask("첨부: 김철수_이력서.pdf") == "첨부: <KR_NAME>_이력서.pdf"
    long_text = "오늘 날씨가 맑습니다. " * 200 + "이상 씨가 서류를 제출했습니다."
    findings = guard.analyze(long_text)
    assert any(f.entity == "KR_NAME" and f.text == "이상"
               and f.start == long_text.rindex("이상") for f in findings)
