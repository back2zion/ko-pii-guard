"""Name/address integration without downloading or importing a neural model.

FakeNER checks the adapter contract, not the accuracy of a trained model. The
curated fixture has independently assigned spans and a separate free-prose track.
"""

import json
from pathlib import Path

import pytest
from presidio_analyzer import RecognizerResult

from ko_pii_guard import DEFAULT_ENTITIES, SUPPORTED_ENTITIES, KoreanPIIGuard
from ko_pii_guard.names import recognize_name_fields

DATA = Path(__file__).resolve().parents[1] / "benchmarks/data/name_address.jsonl"
STRUCTURED_CASES = [
    row for line in DATA.read_text(encoding="utf-8").splitlines()
    if (row := json.loads(line))["track"] == "structured" and row["category"] != "address"
]
FIELD_BOUNDARY_CASES = [case for line in
                        DATA.with_name("name_field_boundaries.jsonl").read_text(
                            encoding="utf-8").splitlines()
                        if (case := json.loads(line))["track"] == "structured"]


def extended_guard(**kwargs):
    kwargs.setdefault("entities", SUPPORTED_ENTITIES)
    return KoreanPIIGuard(**kwargs)


def test_names_and_addresses_require_explicit_opt_in_even_with_backend():
    text = "성명: 오지은, 주소: 서울 강남구 가상누리로 9999"
    backend = FakeNER([span(text, "오지은")])
    guard = KoreanPIIGuard(ner=backend)
    assert "KR_NAME" not in DEFAULT_ENTITIES and "KR_ADDRESS" not in DEFAULT_ENTITIES
    assert guard.analyze(text) == []
    assert guard.mask(text) == text
    assert backend.calls == []


def test_filename_mask_policy_preserves_detection_and_other_masks():
    text = "첨부: 김철수_이력서.pdf, 연락처 010-1234-5678"
    backend = FakeNER([span(text, "김철수")])
    guard = extended_guard(ner=backend)
    assert [f.entity for f in guard.analyze(text)] == ["KR_NAME", "PHONE_NUMBER"]
    assert guard.contains_pii(text)
    assert guard.mask(text) == "첨부: <KR_NAME>_이력서.pdf, 연락처 <PHONE_NUMBER>"
    filename_start, filename_end = text.index("김철수"), text.index(",")
    assert guard.mask(text, should_mask=lambda f: not (
        filename_start <= f.start < filename_end
    )) == "첨부: 김철수_이력서.pdf, 연락처 <PHONE_NUMBER>"


class FakeNER:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def analyze(self, text):
        self.calls.append(text)
        return self.results


def span(text, value, entity="KR_NAME", score=0.95):
    start = text.index(value)
    return RecognizerResult(entity, start, start + len(value), score)


@pytest.fixture(scope="module")
def guard():
    return extended_guard()


@pytest.mark.parametrize("case", [*STRUCTURED_CASES, *FIELD_BOUNDARY_CASES],
                         ids=lambda row: row["id"])
def test_structured_name_fixture_and_mixed_fields(guard, case):
    text = case["text"]
    expected = [(e["entity"], e["start"], e["end"]) for e in case["expected"]]
    actual = guard.analyze(text)
    assert [(f.entity, f.start, f.end) for f in actual] == expected
    assert all(f.text == text[f.start:f.end] for f in actual)
    assert guard.contains_pii(text) is bool(expected)
    assert [(r.start, r.end) for r in recognize_name_fields(text)] == [
        (start, end) for entity, start, end in expected if entity == "KR_NAME"
    ]

    tagged = text
    for entity, start, end in reversed(expected):
        tagged = tagged[:start] + f"<{entity}>" + tagged[end:]
    assert guard.mask(text) == tagged
    starred = list(text)
    for _, start, end in expected:
        starred[start:end] = "*" * (end - start)
    assert guard.mask(text, style="stars") == "".join(starred)


@pytest.mark.parametrize("name,expected", [
    ("이상", "이*"), ("소망", "소*"), ("오지은", "오**"),
    ("남궁민수", "남***"), ("Mina Example", "M*** *******"),
])
def test_partial_names_keep_one_character(guard, name, expected):
    assert guard.mask(f"성명: {name}", style="partial") == f"성명: {expected}"


@pytest.mark.parametrize("value", ["안나·", "안나··가온", "안나·123", "안나·_가온"])
def test_incomplete_middle_dot_names_do_not_return_a_partial_name(guard, value):
    assert guard.analyze(f"성명: {value}") == []


def test_unassigned_role_is_not_a_global_name_denylist(guard):
    assert guard.analyze("담당자: 미정") == []
    assert guard.mask("성명: 미정") == "성명: <KR_NAME>"
    assert guard.mask("담당자: 김미정") == "담당자: <KR_NAME>"


@pytest.mark.parametrize("name", ["Nguyễn An", "Łukasz Test", "Jose\u0301 Test"])
def test_structured_unicode_name_fields_keep_original_spans(guard, name):
    text = f"성명: {name}"
    [finding] = guard.analyze(text)
    assert finding.entity == "KR_NAME"
    assert finding.text == name == text[finding.start:finding.end]
    assert guard.mask(text) == "성명: <KR_NAME>"


def test_partial_addresses_retain_no_letters_or_digits(guard):
    address = "서울특별시 강남구 가상누리로 9999, 테스트아파트 101동 202호"
    text = f"성명: 오지은, 주소: {address}"
    expected_address = "".join("*" if char.isalnum() else char for char in address)
    assert guard.mask(text, style="partial") == f"성명: 오**, 주소: {expected_address}"


@pytest.mark.parametrize("requested,expected", [
    (["KR_NAME"], ["KR_NAME"]),
    (["KR_ADDRESS"], ["KR_ADDRESS"]),
    (["KR_NAME", "KR_ADDRESS"], ["KR_NAME", "KR_ADDRESS"]),
])
def test_entity_filters_for_name_and_address(requested, expected):
    text = "성명: 오지은, 주소: 서울특별시 강남구 가상누리로 9999, 연락처 010-1234-5678"
    guard = extended_guard(entities=requested)
    assert [f.entity for f in guard.analyze(text)] == expected


@pytest.mark.parametrize("text,name", [
    ("신입 이상은 서류를 제출했다.", "이상"),
    ("동료 소망은 새 팀으로 옮겼다.", "소망"),
    ("오지은은 안내문을 작성했다.", "오지은"),
    ("더 이상 변경할 내용은 없습니다.", None),
    ("올해의 소망은 건강입니다.", None),
])
def test_optional_adapter_uses_contextual_spans_without_word_blacklist(text, name):
    # These are injected outputs, not claims about a real model's predictions.
    results = [] if name is None else [span(text, name)]
    backend = FakeNER(results)
    guard = extended_guard(entities=["KR_NAME"], ner=backend)
    assert [f.text for f in guard.analyze(text)] == ([] if name is None else [name])
    assert backend.calls == [text]
    if name:
        expected = text.replace(name, "<KR_NAME>", 1)
        assert guard.mask(text) == expected
    else:
        assert guard.mask(text) == text


def test_optional_ner_is_not_called_for_unrelated_entity_filter():
    class MustNotRun:
        def analyze(self, text):
            raise AssertionError("Name/address NER should not run for a phone-only request")

    guard = extended_guard(entities=["PHONE_NUMBER"], ner=MustNotRun())
    assert guard.mask("연락처 010-1234-5678") == "연락처 <PHONE_NUMBER>"


def test_ner_receives_normalized_text_and_returns_original_offsets():
    text = "동료 Mi\u200bna Example이 문서를 작성했다."
    normalized = "동료 Mina Example이 문서를 작성했다."
    backend = FakeNER([span(normalized, "Mina Example")])
    guard = extended_guard(entities=["KR_NAME"], ner=backend)
    [finding] = guard.analyze(text)
    assert backend.calls == [normalized]
    assert finding.text == "Mi\u200bna Example"
    assert finding.text == text[finding.start:finding.end]
    assert guard.mask(text) == "동료 <KR_NAME>이 문서를 작성했다."
    assert guard.mask(text, style="stars") == (
        "동료 " + "*" * len(finding.text) + "이 문서를 작성했다."
    )


@pytest.mark.parametrize("entity,value,label", [
    ("KR_RRN", "900101-1234568", "주민번호"),
    ("PHONE_NUMBER", "010-1234-5678", "연락처"),
    ("KR_ACCOUNT", "3333-01-1234567", "계좌번호"),
])
def test_structured_identifiers_keep_precedence_over_ner(entity, value, label):
    text = f"{label}: {value}"
    backend = FakeNER([span(text, value, "KR_NAME", 1.0)])
    guard = extended_guard(ner=backend)
    assert [(f.entity, f.text) for f in guard.analyze(text)] == [(entity, value)]
    assert guard.mask(text) == f"{label}: <{entity}>"


def test_explicit_name_field_keeps_full_name_over_shorter_ner_span():
    text = "성명: 남궁민수"
    backend = FakeNER([span(text, "민수", score=1.0)])
    guard = extended_guard(ner=backend)
    assert [(f.entity, f.text) for f in guard.analyze(text)] == [("KR_NAME", "남궁민수")]


def test_full_rule_address_wins_over_name_like_road_token():
    address = "서울특별시 강남구 가상누리로 9999 101동 202호"
    text = f"주소: {address}"
    backend = FakeNER([span(text, "가상누리", score=1.0)])
    guard = extended_guard(ner=backend)
    assert [(f.entity, f.text) for f in guard.analyze(text)] == [("KR_ADDRESS", address)]


def test_rejected_low_score_address_does_not_suppress_a_name():
    text = "소망로 9999에서 만나요."
    backend = FakeNER([
        span(text, "소망", "KR_NAME", 0.95),
        span(text, "소망로 9999", "KR_ADDRESS", 0.1),
    ])
    guard = extended_guard(ner=backend)
    assert [(f.entity, f.text) for f in guard.analyze(text)] == [("KR_NAME", "소망")]


def test_unrequested_address_does_not_suppress_requested_name():
    text = "소망로 9999에서 만나요."
    backend = FakeNER([
        span(text, "소망", "KR_NAME", 0.95),
        span(text, "소망로 9999", "KR_ADDRESS", 1.0),
    ])
    guard = extended_guard(entities=["KR_NAME"], ner=backend)
    assert [(f.entity, f.text) for f in guard.analyze(text)] == [("KR_NAME", "소망")]


def test_accepted_model_address_has_precedence_over_overlapping_name():
    text = "가상누리로 9999에서 만나요."
    backend = FakeNER([
        span(text, "가상누리로 9999", "KR_ADDRESS", 0.9),
        span(text, "가상누리", "KR_NAME", 0.99),
    ])
    guard = extended_guard(ner=backend)
    assert [(f.entity, f.text) for f in guard.analyze(text)] == [
        ("KR_ADDRESS", "가상누리로 9999")
    ]


def test_address_rejected_by_numeric_precedence_does_not_hide_a_separate_name():
    text = "소망은 연락처 010-1234-5678를 남겼다."
    backend = FakeNER([
        span(text, "소망", "KR_NAME", 0.95),
        RecognizerResult("KR_ADDRESS", 0, len(text), 0.9),
    ])
    guard = extended_guard(ner=backend)
    assert [(f.entity, f.text) for f in guard.analyze(text)] == [
        ("KR_NAME", "소망"), ("PHONE_NUMBER", "010-1234-5678")
    ]
    assert guard.mask(text) == "<KR_NAME>은 연락처 <PHONE_NUMBER>를 남겼다."


def test_below_threshold_rule_does_not_suppress_qualified_model_span():
    text = "성명: 오지은"
    backend = FakeNER([span(text, "오지은", score=0.95)])
    guard = extended_guard(ner=backend, score_threshold=0.9)
    assert [(f.entity, f.text, f.score) for f in guard.analyze(text)] == [
        ("KR_NAME", "오지은", 0.95)
    ]


@pytest.mark.parametrize("score,expected", [(0.79, []), (0.8, ["소망"])])
def test_guard_threshold_applies_to_model_candidates(score, expected):
    text = "동료 소망은 새 팀으로 옮겼다."
    backend = FakeNER([span(text, "소망", score=score)])
    guard = extended_guard(ner=backend, score_threshold=0.8)
    assert [f.text for f in guard.analyze(text)] == expected


def test_ner_failure_aborts_masking_instead_of_returning_partial_output():
    class FailingNER:
        def analyze(self, text):
            raise RuntimeError("local model inference failed")

    guard = extended_guard(ner=FailingNER())
    with pytest.raises(RuntimeError, match="local model inference failed"):
        guard.mask("소망의 연락처는 010-1234-5678입니다.")


@pytest.mark.parametrize("text", [
    "담당자 확인 부탁드립니다.", "성명 입력 필요", "예금주 확인 요청",
    "수령인 확인 후 배송", "이름 변경 신청",
])
def test_name_field_words_in_instructions_are_not_person_values(guard, text):
    assert guard.analyze(text) == []


def test_name_does_not_consume_a_following_field_label(guard):
    text = "성명: 오지은 연락처: 미기재"
    assert [(f.entity, f.text) for f in guard.analyze(text)] == [("KR_NAME", "오지은")]
    assert guard.mask(text) == "성명: <KR_NAME> 연락처: 미기재"


@pytest.mark.parametrize("text,candidate", [
    ("상품 이름: 안나·가온", "안나·가온"),
    ("파일 이름: 안나·가온", "안나·가온"),
    ("회사명: 가상테크", "가상테크"),
    ("상품명: 소망", "소망"),
    ("프로젝트명: 오지은", "오지은"),
    ("서비스명: Mina Example", "Mina Example"),
    ("주소: 서울", "서울"),
    ("배송지: 가상동", "가상동"),
    ("성명: 미기재", "미기재"),
    ("성명: {{name}}", "{{name}}"),
    ("성명: 입력 필요", "입력 필요"),
    ("소망이라는 상품을 주문했다.", "소망"),
    ("모델 출력은 ... 이었다.", "..."),
])
def test_explicit_non_person_context_vetoes_high_confidence_model_name(text, candidate):
    backend = FakeNER([span(text, candidate, score=0.99)])
    guard = extended_guard(entities=["KR_NAME"], ner=backend)
    assert guard.analyze(text) == []
    assert guard.mask(text) == text


@pytest.mark.parametrize("text,name", [
    ("소망이라는 사람이 새로 왔다.", "소망"),
    ("이상이라는 동료가 서류를 제출했다.", "이상"),
    ("파일명: 오지은", "오지은"),
    ("오지은이라는 파일을 열었다.", "오지은"),
    ("오지은은 소망이라는 파일을 열었다.", "오지은"),
    ("소망은 이상이라는 상품을 만들었다.", "소망"),
    ("회사명: 가상테크; 동료 소망이 서류를 제출했다.", "소망"),
    ("동료 Nguyễn An이 소개를 마쳤다.", "Nguyễn An"),
    ("동료 Łukasz Test가 소개를 마쳤다.", "Łukasz Test"),
    ("동료 Jose\u0301 Test가 소개를 마쳤다.", "Jose\u0301 Test"),
])
def test_contextual_veto_does_not_reject_person_use_or_unfamiliar_spelling(text, name):
    backend = FakeNER([span(text, name, score=0.99)])
    guard = extended_guard(entities=["KR_NAME"], ner=backend)
    assert [(f.entity, f.text) for f in guard.analyze(text)] == [("KR_NAME", name)]


@pytest.mark.parametrize("place", ["서울", "서울특별시 강남구", "가상동"])
def test_contextual_address_policy_rejects_bare_places(place):
    text = f"{place}에서 만나요."
    backend = FakeNER([span(text, place, "KR_ADDRESS", 0.99)])
    guard = extended_guard(entities=["KR_ADDRESS"], ner=backend)
    assert guard.analyze(text) == []
    assert guard.mask(text) == text


@pytest.mark.parametrize("entity,start,end,score", [
    ("PHONE_NUMBER", 0, 2, 0.9),
    ("KR_NAME", -1, 2, 0.9),
    ("KR_NAME", 0, 0, 0.9),
    ("KR_NAME", 3, 2, 0.9),
    ("KR_NAME", 0, 50, 0.9),
    ("KR_NAME", 0.5, 2, 0.9),
    ("KR_NAME", 0, 2.5, 0.9),
    ("KR_NAME", False, 2, 0.9),
    ("KR_NAME", 0, 2, -0.1),
    ("KR_NAME", 0, 2, 1.1),
    ("KR_NAME", 0, 2, float("nan")),
    ("KR_ADDRESS", 0, 2, float("inf")),
])
def test_invalid_ner_results_fail_before_partial_masking(entity, start, end, score):
    backend = FakeNER([RecognizerResult(entity, start, end, score)])
    guard = extended_guard(ner=backend)
    with pytest.raises(ValueError, match="invalid name/address span"):
        guard.mask("연락처 010-1234-5678, 오지은")
