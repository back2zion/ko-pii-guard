"""Korean PII detection and masking built on Microsoft Presidio.

Presidio already ships recognizers for Korean identifiers, but they are disabled
by default, need an NLP engine configured for Korean, and ignore Korean context
words. This module wires them together so they work in one line:

    >>> from ko_pii_guard import KoreanPIIGuard
    >>> guard = KoreanPIIGuard()
    >>> guard.mask("연락처는 010-1234-5678 입니다")
    '연락처는 <PHONE_NUMBER> 입니다'
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
    RecognizerRegistry,
    RecognizerResult,
)
from presidio_analyzer.nlp_engine import NoOpNlpEngine
from presidio_analyzer.predefined_recognizers import (
    CreditCardRecognizer,
    EmailRecognizer,
    KrBrnRecognizer,
    KrDriverLicenseRecognizer,
    KrFrnRecognizer,
    KrPassportRecognizer,
    KrRrnRecognizer,
    PhoneRecognizer,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from ko_pii_guard.context import (
    KOREAN_CONTEXT,
    boost_with_korean_context,
    require_context_for_undelimited,
)

LANGUAGE = "ko"

DEFAULT_ENTITIES: tuple[str, ...] = (
    "KR_RRN",
    "KR_FRN",
    "KR_BRN",
    "KR_DRIVER_LICENSE",
    "KR_PASSPORT",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "CREDIT_CARD",
)

MaskStyle = Literal["tag", "stars", "partial"]

# Characters kept visible by ``mask(style="partial")``.
_PARTIAL_KEEP = {"KR_RRN": 6, "KR_FRN": 6, "PHONE_NUMBER": 3}


@dataclass(frozen=True)
class Finding:
    """A detected PII span."""

    entity: str
    start: int
    end: int
    score: float
    text: str


# python-phonenumbers treats some 010 ranges as invalid, but any
# 010-XXXX-XXXX string should be masked. This pattern catches Korean mobile
# numbers regardless of allocation status.
KR_MOBILE_PATTERN = Pattern(
    "KR mobile",
    r"(?<!\d)01[016789][-. ]?\d{3,4}[-. ]?\d{4}(?!\d)",
    0.5,
)


def _build_analyzer() -> AnalyzerEngine:
    registry = RecognizerRegistry(supported_languages=[LANGUAGE])
    for recognizer in (
        KrRrnRecognizer(supported_language=LANGUAGE),
        KrFrnRecognizer(supported_language=LANGUAGE),
        KrBrnRecognizer(supported_language=LANGUAGE),
        KrDriverLicenseRecognizer(supported_language=LANGUAGE),
        KrPassportRecognizer(supported_language=LANGUAGE),
        PhoneRecognizer(
            supported_language=LANGUAGE,
            supported_regions=("KR",),
            context=list(KOREAN_CONTEXT["PHONE_NUMBER"]),
        ),
        PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[KR_MOBILE_PATTERN],
            supported_language=LANGUAGE,
            name="KrMobileRecognizer",
        ),
        EmailRecognizer(supported_language=LANGUAGE),
        CreditCardRecognizer(supported_language=LANGUAGE),
    ):
        registry.add_recognizer(recognizer)

    nlp_engine = NoOpNlpEngine(models=[{"lang_code": LANGUAGE, "model_name": "no_op"}])
    nlp_engine.load()
    return AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=[LANGUAGE],
    )


# On equal scores, prefer the more specific Korean identifier. A 13-digit RRN
# can also pass the Luhn check and look like a card number.
_TIE_PRIORITY = {entity: i for i, entity in enumerate(DEFAULT_ENTITIES)}


def _remove_overlaps(results: list[RecognizerResult]) -> list[RecognizerResult]:
    """Keep the highest-scoring result when spans overlap."""
    kept: list[RecognizerResult] = []
    ordered = sorted(
        results,
        key=lambda r: (-r.score, _TIE_PRIORITY.get(r.entity_type, 99), r.start, r.end),
    )
    for result in ordered:
        if all(result.end <= k.start or result.start >= k.end for k in kept):
            kept.append(result)
    return sorted(kept, key=lambda r: r.start)


class KoreanPIIGuard:
    """Detect and mask Korean personal information.

    Args:
        entities: Entity types to detect. Defaults to all supported types.
        score_threshold: Minimum confidence (0-1) for a span to count as PII.
            Korean context words (e.g. "주민번호", "연락처") near a match raise
            its confidence.
    """

    def __init__(
        self,
        entities: Iterable[str] | None = None,
        score_threshold: float = 0.4,
    ):
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1")
        self.entities: list[str] = list(DEFAULT_ENTITIES) if entities is None else list(entities)
        if not self.entities:
            raise ValueError("entities must not be empty; pass None for all types")
        unknown = set(self.entities) - set(DEFAULT_ENTITIES)
        if unknown:
            raise ValueError(f"Unsupported entities: {sorted(unknown)}")
        self.score_threshold = score_threshold
        self._analyzer = _build_analyzer()
        self._anonymizer = AnonymizerEngine()

    def _raw_results(self, text: str) -> list[RecognizerResult]:
        results = self._analyzer.analyze(
            text=text,
            language=LANGUAGE,
            entities=self.entities,
            score_threshold=0.0,
        )
        boosted = require_context_for_undelimited(text, boost_with_korean_context(text, results))
        filtered = [r for r in boosted if r.score >= self.score_threshold]
        return _remove_overlaps(filtered)

    def analyze(self, text: str) -> list[Finding]:
        """Return PII spans found in ``text``, ordered by position."""
        return [
            Finding(r.entity_type, r.start, r.end, round(r.score, 3), text[r.start : r.end])
            for r in self._raw_results(text)
        ]

    def contains_pii(self, text: str) -> bool:
        """Return True if ``text`` contains any PII above the threshold."""
        return bool(self._raw_results(text))

    def mask(self, text: str, style: MaskStyle = "tag") -> str:
        """Return ``text`` with PII replaced.

        Styles:
            ``tag``: ``<KR_RRN>``
            ``stars``: every character replaced with ``*``
            ``partial``: keep a short prefix: 6 characters for RRN/FRN
                (``900101-*******``), 3 for phone numbers (``010-****-****``),
                and 2 for everything else
        """
        if style not in ("tag", "stars", "partial"):
            raise ValueError(f"Unknown mask style: {style}")
        results = self._raw_results(text)
        if not results:
            return text
        if style == "partial":
            return self._partial_mask(text, results)
        if style == "tag":
            operators = {
                r.entity_type: OperatorConfig("replace", {"new_value": f"<{r.entity_type}>"})
                for r in results
            }
        elif style == "stars":
            operators = {
                r.entity_type: OperatorConfig(
                    "mask", {"masking_char": "*", "chars_to_mask": 1000, "from_end": False}
                )
                for r in results
            }
        return self._anonymizer.anonymize(
            text=text, analyzer_results=results, operators=operators
        ).text

    @staticmethod
    def _partial_mask(text: str, results: Sequence[RecognizerResult]) -> str:
        out = []
        cursor = 0
        for r in results:
            span = text[r.start : r.end]
            keep = _PARTIAL_KEEP.get(r.entity_type, 2)
            kept = 0
            masked = []
            for ch in span:
                if ch.isalnum() and kept >= keep:
                    masked.append("*")
                else:
                    if ch.isalnum():
                        kept += 1
                    masked.append(ch)
            out.append(text[cursor : r.start])
            out.append("".join(masked))
            cursor = r.end
        out.append(text[cursor:])
        return "".join(out)
