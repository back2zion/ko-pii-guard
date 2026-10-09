"""Mask every detected character while preserving text between findings."""

import pytest

from ko_pii_guard import KoreanPIIGuard


@pytest.mark.parametrize("separator", [" ", "  ", "\t", "\n", ", ", " / "])
@pytest.mark.parametrize("style", ["tag", "stars", "partial"])
def test_adjacent_phone_findings_preserve_separator(separator, style):
    guard = KoreanPIIGuard(entities=["PHONE_NUMBER"])
    text = f"연락처 010-1234-5678{separator}010-5678-1234 확인"
    assert [f.text for f in guard.analyze(text)] == ["010-1234-5678", "010-5678-1234"]
    replacement = {
        "tag": "<PHONE_NUMBER>",
        "stars": "*" * 13,
        "partial": "010-****-****",
    }[style]
    assert guard.mask(text, style=style) == f"연락처 {replacement}{separator}{replacement} 확인"


def test_stars_mask_all_characters_of_long_detected_email():
    # Recognition currently accepts long email-shaped strings. Mask every
    # detected character even when the value exceeds normal email length limits.
    email = "a" * 1001 + "@example.org"
    text = f"메일 {email} 확인"
    guard = KoreanPIIGuard(entities=["EMAIL_ADDRESS"])
    assert [f.text for f in guard.analyze(text)] == [email]
    masked = guard.mask(text, style="stars")
    assert masked == f"메일 {'*' * len(email)} 확인"
    assert len(masked) == len(text)


def test_mixed_adjacent_entities_keep_punctuation():
    guard = KoreanPIIGuard()
    text = "메일 person@example.org, 연락처 010-1234-5678."
    assert guard.mask(text) == "메일 <EMAIL_ADDRESS>, 연락처 <PHONE_NUMBER>."


@pytest.mark.parametrize("style", ["tag", "stars", "partial"])
def test_mask_policy_receives_original_findings_after_unicode_normalization(style):
    phone = "０１０\u200b－１２３４－５６７８"
    text = f"📎 성명: 오지은, 연락처 {phone}"
    guard = KoreanPIIGuard(entities=["KR_NAME", "PHONE_NUMBER"])
    findings = guard.analyze(text)
    assert [(f.entity, f.text) for f in findings] == [
        ("KR_NAME", "오지은"), ("PHONE_NUMBER", phone),
    ]
    seen = []

    def phone_only(finding):
        seen.append(finding)
        assert finding.text == text[finding.start:finding.end]
        return finding.entity == "PHONE_NUMBER"

    replacement = {
        "tag": "<PHONE_NUMBER>",
        "stars": "*" * len(phone),
        "partial": "０１０\u200b－****－****",
    }[style]
    assert guard.mask(text, style=style, should_mask=phone_only) == (
        f"📎 성명: 오지은, 연락처 {replacement}"
    )
    assert seen == findings
    assert guard.mask(text, style=style, should_mask=lambda finding: False) == text
    assert guard.analyze(text) == findings
    assert guard.contains_pii(text)


@pytest.mark.parametrize("style", ["tag", "stars", "partial"])
def test_mask_policy_errors_propagate_instead_of_returning_unmasked_text(style):
    text = "성명: 오지은, 연락처 010-1234-5678"
    guard = KoreanPIIGuard(entities=["KR_NAME", "PHONE_NUMBER"])
    findings = guard.analyze(text)
    failure = RuntimeError("replacement policy unavailable")

    def fail_on_second_finding(finding):
        if finding.entity == "PHONE_NUMBER":
            raise failure
        return True

    with pytest.raises(RuntimeError) as caught:
        guard.mask(text, style=style, should_mask=fail_on_second_finding)
    assert caught.value is failure
    assert guard.analyze(text) == findings
