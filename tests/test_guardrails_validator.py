import random

import pytest
import synthetic as s

guardrails = pytest.importorskip("guardrails")

from guardrails import Guard  # noqa: E402

from ko_pii_guard.guardrails_validator import KoreanPII  # noqa: E402


def test_fix_masks_pii():
    rng = random.Random(7)
    guard = Guard().use(KoreanPII(on_fail="fix"))
    text = f"주민번호 {s.rrn(rng)} 확인 부탁드립니다"
    result = guard.validate(text)
    assert result.validated_output == "주민번호 <KR_RRN> 확인 부탁드립니다"


def test_passes_clean_text():
    guard = Guard().use(KoreanPII(on_fail="exception"))
    result = guard.validate("개인정보가 없는 문장입니다")
    assert result.validation_passed


def test_exception_on_pii():
    rng = random.Random(8)
    guard = Guard().use(KoreanPII(on_fail="exception"))
    with pytest.raises(Exception, match="KR_RRN"):
        guard.validate(f"주민번호 {s.rrn(rng)}")
