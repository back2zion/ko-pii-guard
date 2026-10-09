import importlib.metadata
import socket

import pytest

import ko_pii_guard
from ko_pii_guard import KoreanPIIGuard
from ko_pii_guard.normalization import normalize_text


@pytest.mark.parametrize("text", [
    "입금 계좌 ３３３３－０１－１２３４５６７",
    "입금 계좌 3333−01−1234567",
    "입금 계좌 3333\u200b\u200d-01-1234567",
    "👤 입금 계좌 3333-01-123\u200b4567 로 보내세요",
])
def test_normalization_preserves_original_spans(text):
    guard = KoreanPIIGuard()
    [finding] = guard.analyze(text)
    assert finding.entity == "KR_ACCOUNT"
    assert text[finding.start:finding.end] == finding.text
    assert guard.mask(text, style="stars") == (
        text[:finding.start] + "*" * len(finding.text) + text[finding.end:]
    )


def test_normalization_does_not_change_unrelated_prose():
    text = "① 상품 · 한\u200b글 · 10㎏"
    assert normalize_text(text).text == text
    assert KoreanPIIGuard().mask(text) == text


def test_long_non_identifier_invisible_run_is_preserved():
    text = "한" + "\u200b" * 10000 + "글"
    result = normalize_text(text)
    assert result.text == text
    assert result.positions is None


def test_normalization_can_be_disabled():
    text = "계좌번호 3333\u200b011234567"
    assert KoreanPIIGuard().contains_pii(text)
    assert not KoreanPIIGuard(normalize_unicode=False).contains_pii(text)


def test_version_matches_distribution():
    assert ko_pii_guard.__version__ == importlib.metadata.version("ko-pii-guard")


def test_email_detection_requires_no_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("PII detection must not open a network connection")

    monkeypatch.setattr(socket, "socket", fail)
    guard = KoreanPIIGuard()
    assert guard.mask("메일 user@example.co.kr") == "메일 <EMAIL_ADDRESS>"


def test_documented_four_group_account_and_context():
    guard = KoreanPIIGuard(entities=["KR_ACCOUNT"])
    assert guard.mask("입금은 농협 302-1234-5678-91 로 부탁드립니다") == (
        "입금은 농협 <KR_ACCOUNT> 로 부탁드립니다"
    )
    assert guard.mask("번호 302-1234-5678-91") == "번호 302-1234-5678-91"
    assert guard.mask("은행 주문번호 302-1234-5678-91") == "은행 주문번호 302-1234-5678-91"
    assert guard.mask("계좌 999-1234-5678-91") == "계좌 999-1234-5678-91"
