"""Character decoder contracts, distinct from trained-model accuracy."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ko_pii_guard.ner import KoreanNER


def test_character_mapping_supports_subtoken_boundaries_and_overlapping_offsets():
    torch = pytest.importorskip("torch")
    from ko_pii_guard.name_context import character_features

    hidden = torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    features, chars, positions = character_features(" 이상의", hidden, [(0, 0), (1, 4), (1, 2)])
    assert features.tolist() == [[0.0, 0.0], [30.0, 31.0], [20.0, 21.0], [20.0, 21.0]]
    assert chars.tolist() == [ord(c) % 4096 for c in " 이상의"]
    assert positions[1].tolist() == [0.0, 0.0]
    assert positions[2].tolist() == pytest.approx([1 / 3, 1 / 3])


def test_character_head_can_end_name_inside_a_token_and_preserve_model_addresses():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        is_fast = True
        model_max_length = 8

        def __call__(self, text, **kwargs):
            # "이상의" is one token; the correct name ends inside it.
            return dict(
                input_ids=torch.tensor([[0, 1, 2, 3, 0]]),
                attention_mask=torch.ones((1, 5), dtype=torch.long),
                offset_mapping=torch.tensor([[[0, 0], [0, 3], [4, 7], [8, 10], [0, 0]]]),
            )

    class Model:
        config = SimpleNamespace(
            max_position_embeddings=8,
            id2label={0: "O", 1: "S-private_person", 2: "S-private_address"},
        )

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            assert kwargs["output_hidden_states"] is True
            logits = torch.tensor(
                [
                    [
                        [10.0, 0.0, 0.0],
                        [0.0, 10.0, 0.0],
                        [10.0, 0.0, 0.0],
                        [0.0, 0.0, 10.0],
                        [10.0, 0.0, 0.0],
                    ]
                ]
            )
            return SimpleNamespace(logits=logits, hidden_states=(torch.zeros((1, 5, 768)),))

    class Head:
        def __call__(self, features, characters, positions, lengths):
            labels = [1, 3, 0, 0, 0, 0, 0, 0, 0, 0]
            logits = torch.full((1, 10, 5), -10.0)
            for i, label in enumerate(labels):
                logits[0, i, label] = 10.0
            return logits

    ner = KoreanNER(Tokenizer(), Model(), stride=2, name_context_head=Head())
    results = ner.analyze("이상의 의견은 길1")
    assert [(r.entity_type, r.start, r.end) for r in results] == [
        ("KR_NAME", 0, 2),
        ("KR_ADDRESS", 8, 10),
    ]


def test_name_head_checkpoint_rejects_mismatched_base_before_reading_weights(tmp_path):
    pytest.importorskip("torch")
    from ko_pii_guard.name_context import load_name_head

    (tmp_path / "name_context_config.json").write_text(json.dumps({"format_version": 100}))
    with pytest.raises(ValueError, match="does not match"):
        load_name_head(tmp_path, model_id="test", revision="test", device="cpu")


def test_training_corpus_has_disjoint_new_names_contexts_and_exact_authored_offsets():
    import runpy

    root = Path(__file__).resolve().parents[1]
    generator = runpy.run_path(str(root / "benchmarks/name_context_training_cases.py"))
    cases = generator["build_cases"]()
    saved = [
        json.loads(line)
        for line in (root / "benchmarks/data/name_context_training.jsonl").read_text().splitlines()
    ]
    assert cases == saved
    assert len({c["id"] for c in cases}) == len(cases)
    for left, right in [
        ("train", "validation"),
        ("train", "evaluation"),
        ("validation", "evaluation"),
    ]:
        for key in ("text", "surface", "context_id"):
            a = {c[key] for c in cases if c["split"] == left and c[key] is not None}
            b = {c[key] for c in cases if c["split"] == right and c[key] is not None}
            assert a.isdisjoint(b)
    for case in cases:
        for span in case["expected"]:
            assert span["entity"] == "KR_NAME"
            assert case["text"][span["start"] : span["end"]] == case["surface"]
    old = [
        json.loads(line)
        for line in (root / "benchmarks/data/name_context.jsonl").read_text().splitlines()
    ]
    development = [c for c in cases if c["id"].startswith("development-")]
    assert len(old) == len(development)
    assert all(c["split"] == "train" for c in development)


@pytest.mark.skipif(
    __import__("os").environ.get("KO_PII_TEST_NAME_CONTEXT") != "1",
    reason="requires optional ner extra and explicitly cached E5 model",
)
def test_trained_character_head_corrects_known_context_and_boundary_failures():
    torch = pytest.importorskip("torch")
    from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard

    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    ner = KoreanNER.from_pretrained(name_context_path=root / "artifacts/name-context-v1")
    guard = KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=ner)
    pairs = [
        ("새해의 소망은 가족 모두의 건강이다.", "새해의 소망은 가족 모두의 건강이다."),
        (
            "조각상의 미소는 보는 방향에 따라 달라 보였다.",
            "조각상의 미소는 보는 방향에 따라 달라 보였다.",
        ),
        (
            "간호사는 대기 중인 환자 사공이든 씨를 진료실로 안내했다.",
            "간호사는 대기 중인 환자 <KR_NAME> 씨를 진료실로 안내했다.",
        ),
        (
            "현장을 목격한 이상이 담당 조사관과 만났다.",
            "현장을 목격한 <KR_NAME>이 담당 조사관과 만났다.",
        ),
    ]
    for text, expected in pairs:
        assert guard.mask(text) == expected
        assert all(f.text == text[f.start : f.end] for f in guard.analyze(text))


@pytest.fixture(scope="module")
def v2_guard():
    torch = pytest.importorskip("torch")
    from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard

    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    return KoreanPIIGuard(
        entities=SUPPORTED_ENTITIES,
        ner=KoreanNER.from_pretrained(
            name_context_path=os.environ.get(
                "KO_PII_TEST_NAME_CONTEXT_PATH", str(root / "artifacts/name-context-v2")
            )
        ),
    )


def inspected_v1_cases():
    root = Path(__file__).resolve().parents[1]
    return [
        case
        for line in (root / "benchmarks/data/name_context_training.jsonl").read_text().splitlines()
        if (case := json.loads(line))["split"] == "evaluation"
    ]


@pytest.mark.parametrize("case", inspected_v1_cases(), ids=lambda case: case["id"])
@pytest.mark.skipif(
    __import__("os").environ.get("KO_PII_TEST_NAME_CONTEXT") != "1",
    reason="requires optional ner extra and explicitly cached E5 model",
)
def test_v2_closes_all_inspected_v1_misses_and_regressions(v2_guard, case):
    # These 140 cases are now development regressions, not new held-out evidence.
    findings = v2_guard.analyze(case["text"])
    assert [(f.entity, f.start, f.end) for f in findings] == [
        (e["entity"], e["start"], e["end"]) for e in case["expected"]
    ], case["id"]
    starred = list(case["text"])
    for e in case["expected"]:
        starred[e["start"] : e["end"]] = "*" * (e["end"] - e["start"])
    assert v2_guard.mask(case["text"], style="stars") == "".join(starred), case["id"]


@pytest.mark.skipif(
    __import__("os").environ.get("KO_PII_TEST_NAME_CONTEXT") != "1",
    reason="requires optional ner extra and explicitly cached E5 model",
)
def test_v2_preserves_multiple_people_filenames_and_late_window_spans(v2_guard):
    text = "김철수 씨와 박영희 씨가 서류를 함께 검토했습니다."
    assert v2_guard.mask(text) == "<KR_NAME> 씨와 <KR_NAME> 씨가 서류를 함께 검토했습니다."
    assert v2_guard.mask("첨부: 김철수_이력서.pdf") == "첨부: <KR_NAME>_이력서.pdf"
    long_text = "오늘 날씨가 맑습니다. " * 200 + "이상 씨가 서류를 제출했습니다."
    findings = v2_guard.analyze(long_text)
    assert [(f.entity, f.start, f.end) for f in findings] == [
        ("KR_NAME", long_text.rindex("이상"), long_text.rindex("이상") + 2)
    ]
    assert findings[0].text == "이상"
