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
