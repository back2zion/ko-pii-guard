"""Bank-specific account evidence must beat incidental Luhn matches."""

import pytest

from ko_pii_guard import KoreanPIIGuard


@pytest.mark.parametrize("text,value", [
    ("카카오뱅크 입금 계좌 3333120145448", "3333120145448"),
    ("농협 입금 계좌 3025382178204", "3025382178204"),
])
def test_documented_undelimited_accounts_win_incidental_card_checksum(text, value):
    guard = KoreanPIIGuard()
    assert [(f.entity, f.text) for f in guard.analyze(text)] == [("KR_ACCOUNT", value)]
    assert guard.mask(text, style="stars") == text.replace(value, "*" * len(value))
    assert [f.entity for f in KoreanPIIGuard(entities=["KR_ACCOUNT"]).analyze(text)] == [
        "KR_ACCOUNT"
    ]


@pytest.mark.parametrize("text,entity", [
    ("카카오뱅크 카드번호 3333120145448", "CREDIT_CARD"),
    ("신한 계좌번호 3333120145448", "CREDIT_CARD"),
    ("카카오뱅크 계좌번호 5555555555554", "CREDIT_CARD"),
])
def test_account_override_requires_matching_bank_layout_and_explicit_label(text, entity):
    assert [f.entity for f in KoreanPIIGuard().analyze(text)] == [entity]
