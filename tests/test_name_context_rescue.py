"""A rescue can fill a missed span but cannot displace accepted names or addresses."""

import pytest
from presidio_analyzer import RecognizerResult


def test_rescue_preserves_original_objects_scores_and_all_overlapping_spans():
    pytest.importorskip("torch")
    from ko_pii_guard.name_context import add_nonoverlapping_name_results

    person = RecognizerResult("KR_NAME", 5, 8, .91)
    address = RecognizerResult("KR_ADDRESS", 12, 20, .92)
    missed = RecognizerResult("KR_NAME", 0, 2, .995)
    candidates = [missed, RecognizerResult("KR_NAME", 4, 9, .999),
                  RecognizerResult("KR_NAME", 6, 7, 1),
                  RecognizerResult("KR_NAME", 12, 15, 1),
                  RecognizerResult("KR_ADDRESS", 21, 25, 1)]
    result = add_nonoverlapping_name_results([person, address], candidates)
    assert result == [missed, person, address]
    assert result[1] is person and result[2] is address
    assert (person.start, person.end, person.score) == (5, 8, .91)
    assert candidates[0] is missed


def test_rescue_deduplicates_and_keeps_adjacent_names():
    pytest.importorskip("torch")
    from ko_pii_guard.name_context import add_nonoverlapping_name_results

    original = RecognizerResult("KR_NAME", 2, 4, .95)
    adjacent = RecognizerResult("KR_NAME", 4, 6, .99)
    result = add_nonoverlapping_name_results([original], [
        RecognizerResult("KR_NAME", 2, 4, 1), adjacent,
        RecognizerResult("KR_NAME", 4, 5, .98),
    ])
    assert result == [original, adjacent]
    assert result[0] is original


def test_real_ner_keeps_old_rescue_and_adds_disjoint_refined_candidate():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from ko_pii_guard.ner import KoreanNER

    class Tokenizer:
        is_fast = True
        model_max_length = 8

        def __call__(self, text, **kwargs):
            return dict(input_ids=torch.tensor([[0, 1, 2, 3, 0]]),
                        attention_mask=torch.ones(1, 5, dtype=torch.long),
                        offset_mapping=torch.tensor([[[0, 0], [0, 2], [3, 5], [6, 8], [0, 0]]]))

    class Model:
        config = SimpleNamespace(max_position_embeddings=8, id2label={0: "O"})

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.zeros(1, 5, 1),
                                   hidden_states=(torch.zeros(1, 5, 768),))

    class Head:
        def __init__(self, labels):
            self.labels = labels

        def __call__(self, features, characters, positions, lengths):
            logits = torch.full((1, 8, 5), -20.)
            for i, label in enumerate(self.labels):
                logits[0, i, label] = 20.
            return logits

    primary = Head([0]*8)
    primary.rescue_head = Head([1, 3, 0, 0, 0, 0, 0, 0])
    primary.rescue_threshold = .99
    # The new head forgot the old full name and proposes a wrong shorter span.
    primary.extra_rescue_heads = [Head([4, 0, 0, 1, 3, 0, 0, 0])]
    primary.extra_rescue_thresholds = [.99]
    ner = KoreanNER(Tokenizer(), Model(), stride=2, name_context_head=primary)
    assert [(r.entity_type, r.start, r.end) for r in ner.analyze("가나 다라 마바")] == [
        ("KR_NAME", 0, 2), ("KR_NAME", 3, 5)
    ]
