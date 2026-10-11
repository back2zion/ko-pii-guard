"""A card number explicitly labeled as a card is masked even when Luhn fails."""

import pytest

from ko_pii_guard import KoreanPIIGuard

guard = KoreanPIIGuard()
BAD_LUHN = "4161 9135 6354 8285"


def cards(text):
    return [(f.entity, f.text) for f in guard.analyze(text)]


@pytest.mark.parametrize("text", [
    f"카드번호: {BAD_LUHN}",
    f"카드 번호는 {BAD_LUHN}입니다",
    f"이번엔 법인 카드 {BAD_LUHN}로 예약했어요",
    f"결제 예정 카드인 {BAD_LUHN}의 잔액을 확인해주세요",
    f"신용카드 {BAD_LUHN.replace(' ', '-')} 로 결제",
    "체크카드번호 4161913563548285 입니다",
])
def test_labeled_card_with_bad_luhn_is_detected(text):
    assert [e for e, _ in cards(text)] == ["CREDIT_CARD"]
    assert guard.mask(text, style="stars").count("*") >= 16


@pytest.mark.parametrize("text", [
    f"주문번호 {BAD_LUHN}",
    f"카드 승인번호 {BAD_LUHN}",
    f"거래번호: {BAD_LUHN} 카드로 결제",
    f"일련번호 {BAD_LUHN} (카드 배송)",
    f"카드를 분실했습니다. 문의 {BAD_LUHN}",
    "카드 이용 내역 확인 후 4161 9135 6354 8285 참조",
    BAD_LUHN,
    f"참조 {BAD_LUHN}",
    "카드번호 4161 9135 6354 828",
    "카드번호 4161 9135 6354 82851",
])
def test_unlabeled_or_other_identifier_bad_luhn_is_kept(text):
    assert cards(text) == []


def test_valid_luhn_card_unchanged():
    assert cards("카드번호 4111 1111 1111 1111") == [("CREDIT_CARD", "4111 1111 1111 1111")]
    assert cards("결제 4111 1111 1111 1111") == [("CREDIT_CARD", "4111 1111 1111 1111")]
