"""Korean context words that raise confidence for nearby matches.

Presidio's context enhancement relies on tokens and lemmas from an NLP model.
Korean text needs a morphological analyzer for that, which is a heavy
dependency. Instead, this module checks for Korean keywords in a fixed window
before each match. The check is simple and needs no model.
"""

from __future__ import annotations

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
UNDELIMITED_NEEDS_CONTEXT = ("KR_DRIVER_LICENSE",)
UNDELIMITED_SCORE = 0.3


def require_context_for_undelimited(
    text: str, results: list[RecognizerResult], window: int = WINDOW
) -> list[RecognizerResult]:
    """Lower the score of bare digit runs that have no Korean context."""
    adjusted: list[RecognizerResult] = []
    for r in results:
        span = text[r.start : r.end]
        needs_context = (r.entity_type in UNDELIMITED_NEEDS_CONTEXT and span.isdigit()) or (
            # A bare digit run like "8217..." is read as +82 by phonenumbers,
            # but domestic Korean numbers start with 0.
            r.entity_type == "PHONE_NUMBER" and span.isdigit() and not span.startswith("0")
        )
        if needs_context:
            before = text[max(0, r.start - window) : r.start]
            keywords = KOREAN_CONTEXT.get(r.entity_type, ())
            if not any(k in before for k in keywords):
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
