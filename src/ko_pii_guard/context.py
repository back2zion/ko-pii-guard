"""Korean context words that raise confidence for nearby matches.

Presidio's context enhancement relies on tokens and lemmas from an NLP model.
Korean text needs a morphological analyzer for that, which is a heavy
dependency. Instead, this module checks for Korean keywords in a fixed window
before each match. The check is simple and needs no model.
"""

from __future__ import annotations

import re

from presidio_analyzer import RecognizerResult

KOREAN_CONTEXT: dict[str, tuple[str, ...]] = {
    "KR_RRN": ("주민등록번호", "주민번호", "주민등록", "생년월일"),
    "KR_FRN": ("외국인등록번호", "외국인등록", "외국인번호"),
    "KR_BRN": ("사업자등록번호", "사업자번호", "사업자등록", "사업자"),
    "KR_DRIVER_LICENSE": ("운전면허번호", "운전면허", "면허번호"),
    "KR_PASSPORT": ("여권번호", "여권"),
    "PHONE_NUMBER": ("전화번호", "휴대폰", "핸드폰", "연락처", "전화", "휴대전화"),
    "EMAIL_ADDRESS": ("이메일", "메일", "전자우편"),
    "CREDIT_CARD": ("카드번호", "신용카드", "체크카드", "카드"),
    "KR_ACCOUNT": (
        "계좌", "계좌번호", "입금", "송금", "이체", "예금주", "통장", "은행",
        "농협", "국민", "신한", "우리", "하나", "기업", "카카오뱅크", "토스뱅크",
        "새마을금고", "우체국",
    ),
}

WINDOW = 20
BOOST = 0.35


def boost_with_korean_context(
    text: str, results: list[RecognizerResult], window: int = WINDOW, boost: float = BOOST
) -> list[RecognizerResult]:
    """Return results with scores raised when Korean context words precede them."""
    boosted: list[RecognizerResult] = []
    for r in results:
        keywords = KOREAN_CONTEXT.get(r.entity_type, ())
        before = text[max(0, r.start - window) : r.start]
        score = r.score
        if keywords and any(k in before for k in keywords):
            score = min(1.0, r.score + boost)
        boosted.append(
            RecognizerResult(
                entity_type=r.entity_type,
                start=r.start,
                end=r.end,
                score=score,
                analysis_explanation=r.analysis_explanation,
                recognition_metadata=r.recognition_metadata,
            )
        )
    return boosted


# Long digit runs without separators are often invoice or order numbers. These
# entities only count when written with separators or preceded by context.
UNDELIMITED_NEEDS_CONTEXT = ("KR_DRIVER_LICENSE", "KR_ACCOUNT")
UNDELIMITED_SCORE = 0.3


def _account_context(text: str, start: int, window: int) -> tuple[bool, bool]:
    """Use the closest label in the current clause to limit context leakage."""
    before = re.split(r"[\n,;!?]", text[max(0, start - window):start])[-1]
    positive = max((before.rfind(k) for k in KOREAN_CONTEXT["KR_ACCOUNT"]), default=-1)
    negative = max((before.rfind(k) for k in (
        "주문번호", "송장번호", "참조번호", "상품코드", "거래번호", "승인번호",
        "주민번호", "주민등록번호", "사업자번호", "사업자등록번호", "카드번호",
        "전화번호", "연락처", "운전면허번호", "날짜",
    )), default=-1)
    return positive >= 0 and positive > negative, negative >= 0 and negative > positive


def require_context_for_undelimited(
    text: str, results: list[RecognizerResult], window: int = WINDOW
) -> list[RecognizerResult]:
    """Lower the score of bare digit runs that have no Korean context."""
    adjusted: list[RecognizerResult] = []
    for r in results:
        span = text[r.start : r.end]
        if r.entity_type == "KR_ACCOUNT":
            has_context, blocked = _account_context(text, r.start, window)
            if blocked or (span.isdigit() and not has_context):
                continue
        needs_context = (r.entity_type in UNDELIMITED_NEEDS_CONTEXT and span.isdigit()) or (
            # A bare digit run like "8217..." is read as +82 by phonenumbers,
            # but domestic Korean numbers start with 0.
            r.entity_type == "PHONE_NUMBER" and span.isdigit() and not span.startswith("0")
        )
        if needs_context:
            before = text[max(0, r.start - window) : r.start]
            keywords = KOREAN_CONTEXT.get(r.entity_type, ())
            if not any(k in before for k in keywords):
                if r.entity_type == "KR_ACCOUNT":
                    continue
                r = RecognizerResult(
                    entity_type=r.entity_type,
                    start=r.start,
                    end=r.end,
                    score=min(r.score, UNDELIMITED_SCORE),
                    analysis_explanation=r.analysis_explanation,
                    recognition_metadata=r.recognition_metadata,
                )
        adjusted.append(r)
    return adjusted
