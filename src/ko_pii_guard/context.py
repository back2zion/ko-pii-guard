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

# Sentence/field boundaries prevent context from a different value leaking in.
_CLAUSE_BOUNDARY = re.compile(r"[\r\n,;.!?。！？]")
_ACCOUNT_BLOCKERS = (
    "주문번호", "주문 ID", "주문ID", "주문", "송장번호", "송장", "참조번호",
    "상품코드", "제품번호", "품번", "거래번호", "승인번호", "버전", "코드",
    "일련번호", "시리얼", "주민번호", "주민등록번호", "사업자번호", "사업자등록번호",
    "카드번호", "전화번호", "연락처", "운전면허번호", "날짜",
)
_AMBIGUOUS_BANK_WORDS = frozenset(("우리", "하나", "기업", "국민"))


def _before(text: str, start: int, window: int) -> str:
    return _CLAUSE_BOUNDARY.split(text[max(0, start - window):start])[-1]


def has_korean_context(text: str, start: int, entity: str, window: int = WINDOW) -> bool:
    """Whether the current clause contains a preceding label for this type."""
    if entity == "KR_ACCOUNT":
        return _account_context(text, start, window)[0]
    before = _before(text, start, window)
    return any(keyword in before for keyword in KOREAN_CONTEXT.get(entity, ()))


def _account_context(text: str, start: int, window: int) -> tuple[bool, bool]:
    """Use the closest label; common bank words need label-like placement."""
    before = _before(text, start, window)
    positive = -1
    for keyword in KOREAN_CONTEXT["KR_ACCOUNT"]:
        for match in re.finditer(re.escape(keyword), before):
            index = match.start()
            if index and (before[index - 1].isascii() and before[index - 1].isalnum()):
                continue
            if keyword in _AMBIGUOUS_BANK_WORDS:
                tail = before[match.end():]
                if index and before[index - 1].isalnum():
                    continue
                if not re.fullmatch(r"(?:은행)?[\s:：=\[\]()]*", tail):
                    continue
            positive = max(positive, index)
    negative = max((before.rfind(k) for k in _ACCOUNT_BLOCKERS), default=-1)
    return positive >= 0 and positive > negative, negative >= 0 and negative > positive


def boost_with_korean_context(
    text: str, results: list[RecognizerResult], window: int = WINDOW, boost: float = BOOST
) -> list[RecognizerResult]:
    """Return results with scores raised when Korean context words precede them."""
    boosted: list[RecognizerResult] = []
    for r in results:
        keywords = KOREAN_CONTEXT.get(r.entity_type, ())
        before = _before(text, r.start, window)
        score = r.score
        has_context = (
            _account_context(text, r.start, window)[0] if r.entity_type == "KR_ACCOUNT"
            else keywords and any(k in before for k in keywords)
        )
        if has_context:
            score = min(1.0, round(r.score + boost, 10))
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


def require_context_for_undelimited(
    text: str, results: list[RecognizerResult], window: int = WINDOW
) -> list[RecognizerResult]:
    """Apply context gates to ambiguous runs and account candidates."""
    adjusted: list[RecognizerResult] = []
    for r in results:
        span = text[r.start : r.end]
        if r.entity_type == "KR_ACCOUNT":
            has_context, blocked = _account_context(text, r.start, window)
            digits = sum(char.isdigit() for char in span)
            if blocked or not 10 <= digits <= 14:
                continue
            if (span.isdigit() or span.count("-") == 3) and not has_context:
                continue
            adjusted.append(r)
            continue
        needs_context = (r.entity_type in UNDELIMITED_NEEDS_CONTEXT and span.isdigit()) or (
            # A bare digit run like "8217..." is read as +82 by phonenumbers,
            # but domestic Korean numbers start with 0.
            r.entity_type == "PHONE_NUMBER" and span.isdigit() and not span.startswith("0")
        )
        if needs_context:
            before = _before(text, r.start, window)
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
