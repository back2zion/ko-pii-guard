"""Guardrails AI validator for Korean PII.

Requires the ``guardrails`` extra: ``pip install "ko-pii-guard[guardrails]"``.

    from guardrails import Guard
    from ko_pii_guard.guardrails_validator import KoreanPII

    guard = Guard().use(KoreanPII(on_fail="fix"))
    guard.validate("주민번호 900101-1234567 확인 부탁드립니다").validated_output
    # '주민번호 <KR_RRN> 확인 부탁드립니다'
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from guardrails.validator_base import (
    ErrorSpan,
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

from ko_pii_guard.analyzer import KoreanPIIGuard


@register_validator(name="back2zion/korean_pii", data_type="string")
class KoreanPII(Validator):
    """Fails when the value contains Korean personal information.

    The programmatic fix replaces each span with ``<ENTITY>`` tags.

    Args:
        entities: Entity types to detect. Defaults to all supported types.
        score_threshold: Minimum confidence for a span to count as PII.
    """

    def __init__(
        self,
        entities: list[str] | None = None,
        score_threshold: float = 0.4,
        on_fail: Callable | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            on_fail=on_fail, entities=entities, score_threshold=score_threshold, **kwargs
        )
        self._guard = KoreanPIIGuard(entities=entities, score_threshold=score_threshold)

    def validate(self, value: Any, metadata: dict = {}) -> ValidationResult:  # noqa: B006
        findings = self._guard.analyze(value)
        if not findings:
            return PassResult()
        entities = sorted({f.entity for f in findings})
        return FailResult(
            error_message=f"Korean PII detected: {', '.join(entities)}",
            fix_value=self._guard.mask(value),
            error_spans=[
                ErrorSpan(start=f.start, end=f.end, reason=f"{f.entity} detected") for f in findings
            ],
        )
