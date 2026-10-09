import random

import pytest
import synthetic as s

from ko_pii_guard import DEFAULT_ENTITIES, KoreanPIIGuard


@pytest.fixture(scope="module")
def guard():
    return KoreanPIIGuard()


@pytest.fixture
def rng():
    return random.Random(42)


def entities(guard, text):
    return [f.entity for f in guard.analyze(text)]


def test_rrn_with_and_without_dash(guard, rng):
    assert entities(guard, f"주민번호 {s.rrn(rng)} 확인") == ["KR_RRN"]
    assert entities(guard, f"번호는 {s.rrn(rng, dash=False)} 입니다") == ["KR_RRN"]


def test_foreigner_registration_number(guard, rng):
    assert entities(guard, f"외국인등록번호 {s.rrn(rng, gender_digit=5)}") == ["KR_FRN"]


def test_business_registration_number(guard, rng):
    assert entities(guard, f"사업자등록번호 {s.brn(rng)}") == ["KR_BRN"]


def test_korean_mobile_numbers(guard, rng):
    assert entities(guard, f"연락처 {s.mobile(rng)} 로 연락") == ["PHONE_NUMBER"]
    assert entities(guard, f"휴대폰 {s.mobile(rng, dash=False)}") == ["PHONE_NUMBER"]


def test_driver_license(guard, rng):
    assert entities(guard, f"운전면허번호 {s.driver_license(rng)}") == ["KR_DRIVER_LICENSE"]


def test_passport_requires_korean_context(guard, rng):
    number = s.passport(rng)
    assert entities(guard, f"여권번호 {number}") == ["KR_PASSPORT"]
    assert entities(guard, f"코드 {number}") == []


def test_email_and_card(guard, rng):
    assert entities(guard, f"메일 {s.email(rng)}") == ["EMAIL_ADDRESS"]
    assert entities(guard, f"카드번호 {s.card(rng)}") == ["CREDIT_CARD"]


@pytest.mark.parametrize(
    "text",
    [
        "오늘 매출은 1,234,567원입니다",
        "회의는 2026-10-08 14:00에 합니다",
        "주문번호 2024010112345 확인 바랍니다",
        "버전 3.11.4로 업데이트했습니다",
    ],
)
def test_no_false_positives_on_common_numbers(guard, text):
    assert guard.analyze(text) == []


def test_multiple_entities_in_one_text(guard, rng):
    text = f"홍길동 고객(주민번호 {s.rrn(rng)}, 연락처 {s.mobile(rng)})의 서류입니다"
    assert entities(guard, text) == ["KR_RRN", "PHONE_NUMBER"]


def test_mask_tag(guard, rng):
    text = f"주민번호 {s.rrn(rng)} 확인"
    assert guard.mask(text) == "주민번호 <KR_RRN> 확인"


def test_mask_stars_keeps_length(guard, rng):
    number = s.mobile(rng)
    masked = guard.mask(f"연락처 {number}", style="stars")
    assert masked == "연락처 " + "*" * len(number)


def test_mask_partial_keeps_prefix(guard, rng):
    number = s.rrn(rng)
    masked = guard.mask(f"주민번호 {number}", style="partial")
    assert masked == f"주민번호 {number[:6]}-*******"


def test_mask_without_pii_returns_input(guard):
    text = "개인정보가 없는 문장입니다"
    assert guard.mask(text) == text
    assert guard.contains_pii(text) is False


def test_entity_filter(rng):
    only_phone = KoreanPIIGuard(entities=["PHONE_NUMBER"])
    text = f"주민번호 {s.rrn(rng)}, 연락처 {s.mobile(rng)}"
    assert entities(only_phone, text) == ["PHONE_NUMBER"]


def test_invalid_arguments():
    with pytest.raises(ValueError):
        KoreanPIIGuard(entities=["US_SSN"])
    with pytest.raises(ValueError):
        KoreanPIIGuard(score_threshold=1.5)
    with pytest.raises(ValueError):
        KoreanPIIGuard().mask("주민번호 900101-1234567", style="unknown")  # type: ignore[arg-type]


def test_default_entities_are_exposed():
    assert "KR_RRN" in DEFAULT_ENTITIES
    assert len(DEFAULT_ENTITIES) == len(set(DEFAULT_ENTITIES))


def test_mobile_range_rejected_by_phonenumbers_is_still_masked(guard):
    # python-phonenumbers marks this range as invalid; it must still be caught.
    assert entities(guard, "연락처 010-5986-0000 로 전화주세요") == ["PHONE_NUMBER"]


def test_bare_twelve_digit_tracking_number_is_not_a_license(guard):
    assert guard.analyze("송장번호 237751597627") == []
    assert entities(guard, "운전면허번호 237751597627") == ["KR_DRIVER_LICENSE"]


def test_digits_starting_with_82_are_not_phone_numbers_without_context(guard):
    assert guard.analyze("송장번호 821775099818") == []


def test_frn_wins_over_phone_interpretation(guard):
    assert entities(guard, "등록번호 820519-6736018") == ["KR_FRN"]


def test_rrn_preferred_over_card_on_tie(guard):
    # This synthetic RRN also passes the Luhn check.
    assert entities(guard, "번호는 6507041449743 입니다") == ["KR_RRN"]


def test_partial_mask_keeps_short_prefix_for_other_types(guard):
    assert guard.mask("연락처 010-0000-0000", style="partial") == "연락처 010-****-****"
    assert (
        guard.mask("사업자등록번호 123-45-67891", style="partial") == "사업자등록번호 12*-**-*****"
    )


def test_unknown_style_rejected_even_without_pii(guard):
    with pytest.raises(ValueError):
        guard.mask("개인정보 없음", style="bogus")  # type: ignore[arg-type]


def test_empty_entities_rejected():
    with pytest.raises(ValueError):
        KoreanPIIGuard(entities=[])


@pytest.mark.parametrize("number", ["110-123-456789", "3333-01-1234567", "3333011234567"])
def test_account_apis(guard, number):
    text = f"계좌번호 {number} 확인"
    findings = guard.analyze(text)
    assert len(findings) == 1
    assert findings[0].entity == "KR_ACCOUNT"
    assert findings[0].text == number
    assert text[findings[0].start:findings[0].end] == number
    assert guard.contains_pii(text)
    assert guard.mask(text) == "계좌번호 <KR_ACCOUNT> 확인"
    assert guard.mask(text, style="stars") == f"계좌번호 {'*' * len(number)} 확인"
    expected = ""
    kept = 0
    for ch in number:
        expected += ch if not ch.isdigit() or kept < 3 else "*"
        kept += ch.isdigit()
    assert guard.mask(text, style="partial") == f"계좌번호 {expected} 확인"


@pytest.mark.parametrize("threshold", [0.0, 0.2, 0.4])
def test_account_context_is_mandatory(threshold):
    guard = KoreanPIIGuard(entities=["KR_ACCOUNT"], score_threshold=threshold)
    text = "코드 77777712345678"
    assert guard.analyze(text) == []
    assert not guard.contains_pii(text)
    assert guard.mask(text) == text


def test_delimited_account_does_not_require_context(guard):
    assert entities(guard, "번호 110-123-456789") == ["KR_ACCOUNT"]


@pytest.mark.parametrize("length", [10, 11, 12, 13, 14, 15, 16])
def test_undelimited_account_without_context_at_every_length(length):
    guard = KoreanPIIGuard(entities=["KR_ACCOUNT"], score_threshold=0)
    assert guard.analyze("코드 " + "7" * length) == []


@pytest.mark.parametrize("dash", [True, False])
def test_accounts_do_not_replace_existing_identifiers(guard, rng, dash):
    cases = [
        (s.rrn(rng, dash=dash), "KR_RRN"),
        (s.brn(rng, dash=dash), "KR_BRN"),
        (s.mobile(rng, dash=dash), "PHONE_NUMBER"),
        (s.card(rng), "CREDIT_CARD"),
        (s.driver_license(rng), "KR_DRIVER_LICENSE"),
    ]
    account_only = KoreanPIIGuard(entities=["KR_ACCOUNT"])
    for number, entity in cases:
        text = f"은행 계좌 확인: {number}"
        assert entities(guard, text) == [entity]
        assert account_only.analyze(text) == []


@pytest.mark.parametrize("number", [
    "2026-10-09", "3333011234567890", "X3333011234567", "X3333-01-1234567",
    "1234-5678-9012-3456", "11-22-123456-78",
])
def test_account_boundaries(number):
    guard = KoreanPIIGuard(entities=["KR_ACCOUNT"])
    assert guard.analyze(f"계좌번호 {number}") == []


@pytest.mark.parametrize("keyword", [
    "계좌", "계좌번호", "입금", "송금", "이체", "예금주", "통장", "은행",
    "농협", "국민", "신한", "우리", "하나", "기업", "카카오뱅크", "토스뱅크",
    "새마을금고", "우체국",
])
def test_account_context_keywords(guard, keyword):
    assert entities(guard, f"{keyword} 3333011234567") == ["KR_ACCOUNT"]


def test_account_context_window_and_direction():
    guard = KoreanPIIGuard(entities=["KR_ACCOUNT"])
    assert guard.analyze("3333011234567 계좌번호") == []
    assert guard.analyze("계좌번호 " + "가" * 21 + " 3333011234567") == []


@pytest.mark.parametrize("text", [
    "은행 주문번호 3333011234567",
    "계좌 안내; 참조번호 3333-01-1234567",
    "은행 안내\n3333011234567",
    "계좌 안내, 3333011234567",
])
def test_account_context_does_not_leak_across_labels_or_clauses(text):
    assert KoreanPIIGuard(entities=["KR_ACCOUNT"]).analyze(text) == []


def test_nearest_account_label_wins(guard):
    assert entities(guard, "주문번호 확인 후 계좌번호 3333011234567") == ["KR_ACCOUNT"]


@pytest.mark.parametrize("bank", s.ACCOUNT_LAYOUTS)
def test_synthetic_account_layout(bank, rng):
    number = s.account(rng, bank)
    assert 10 <= len(number.replace("-", "")) <= 14
    assert len(number.split("-")) == (4 if bank == "농협" else 3)
