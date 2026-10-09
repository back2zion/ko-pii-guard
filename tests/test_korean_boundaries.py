import pytest

from ko_pii_guard import KoreanPIIGuard


@pytest.fixture(scope="module")
def guard():
    return KoreanPIIGuard()


@pytest.mark.parametrize("particle", ["로", "으로", "에서", "를", "을", "님의", "의"])
@pytest.mark.parametrize("value,entity", [
    ("user@example.co.kr", "EMAIL_ADDRESS"),
    ("4111-1111-1111-1111", "CREDIT_CARD"),
])
def test_korean_particles_preserved(guard, particle, value, entity):
    text = f"확인 {value}{particle} 전달"
    [finding] = guard.analyze(text)
    assert finding.entity == entity
    assert finding.text == value
    assert guard.mask(text) == f"확인 <{entity}>{particle} 전달"


@pytest.mark.parametrize("text", [
    "X4111111111111111Z", "4111111111111111-1234", "user@example.invalid",
])
def test_korean_boundary_does_not_allow_ascii_identifier_fragments(guard, text):
    assert guard.analyze(text) == []


def test_impossible_birth_date_requires_explicit_context(guard):
    assert not any(f.entity == "KR_RRN" for f in guard.analyze("번호 900231-1234567"))
    assert guard.mask("주민번호 900231-1234567") == "주민번호 <KR_RRN>"
