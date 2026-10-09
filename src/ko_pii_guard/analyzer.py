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

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

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
from tldextract import TLDExtract

from ko_pii_guard.account_formats import FOUR_GROUP_ACCOUNT_FORMATS
from ko_pii_guard.addresses import recognize_addresses
from ko_pii_guard.context import (
    KOREAN_CONTEXT,
    boost_with_korean_context,
    has_korean_context,
    require_context_for_undelimited,
)
from ko_pii_guard.names import allows_contextual_name, recognize_name_fields
from ko_pii_guard.normalization import NormalizedText, normalize_text
from ko_pii_guard.recognition import collect_results
from ko_pii_guard.spans import SpanIndex, select_non_overlapping
from ko_pii_guard.validation import has_valid_registration_date

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
    "KR_ACCOUNT",
)
SUPPORTED_ENTITIES = (*DEFAULT_ENTITIES, "KR_NAME", "KR_ADDRESS")
_STRUCTURED_ENTITIES = DEFAULT_ENTITIES

MaskStyle = Literal["tag", "stars", "partial"]

# Characters kept visible by ``mask(style="partial")``.
_PARTIAL_KEEP = {
    "KR_RRN": 6, "KR_FRN": 6, "PHONE_NUMBER": 3, "KR_ACCOUNT": 3,
    "KR_NAME": 1, "KR_ADDRESS": 0,
}


class NameAddressNER(Protocol):
    """A local name/address recognizer returning spans in its input text."""

    def analyze(self, text: str) -> list[RecognizerResult]: ...


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
    r"(?<![A-Za-z0-9_-])01[016789][-. ]?\d{3,4}[-. ]?\d{4}(?![A-Za-z0-9_-])",
    0.5,
)

# Do not consume a fragment of a longer digit run or hyphenated identifier.
KR_ACCOUNT_PATTERNS = [
    Pattern("KR account with separators",
            r"(?<![A-Za-z0-9_-])\d{2,6}-\d{2,6}-\d{2,8}(?![A-Za-z0-9_-])", 0.4),
    Pattern("KR account without separators",
            r"(?<![A-Za-z0-9_-])\d{10,14}(?![A-Za-z0-9_-])", 0.1),
]
KR_ACCOUNT_PATTERNS.extend(
    Pattern(layout.name, rf"(?<![A-Za-z0-9_-])(?:{layout.regex})(?![A-Za-z0-9_-])", 0.4)
    for layout in FOUR_GROUP_ACCOUNT_FORMATS
)


class _OfflineEmailRecognizer(EmailRecognizer):
    """Validate against tldextract's bundled suffix snapshot, without I/O."""

    def __init__(self):
        # Keep upstream Unicode/IDN support; the additional ASCII pattern stops
        # before Korean particles rather than absorbing them into the TLD.
        korean_boundary = Pattern(
            "Email before Korean particle",
            r"(?<![A-Za-z0-9.!#$%&'*+/=?^_`{|}~-])"
            r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
            r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+"
            r"[A-Za-z]{2,63}(?![A-Za-z0-9_.@-])",
            0.5,
        )
        super().__init__(supported_language=LANGUAGE,
                         patterns=[*EmailRecognizer.PATTERNS, korean_boundary])
        self._extract = TLDExtract(suffix_list_urls=(), cache_dir=None)

    def validate_result(self, pattern_text: str) -> bool:
        return bool(self._extract(pattern_text).fqdn)


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
        _OfflineEmailRecognizer(),
        CreditCardRecognizer(
            supported_language=LANGUAGE,
            patterns=[Pattern(
                pattern.name,
                r"(?<![A-Za-z0-9_-])" + pattern.regex.removeprefix(r"\b").removesuffix(r"\b")
                + r"(?![A-Za-z0-9_-])",
                pattern.score,
            ) for pattern in CreditCardRecognizer.PATTERNS],
        ),
        PatternRecognizer(
            supported_entity="KR_ACCOUNT",
            patterns=KR_ACCOUNT_PATTERNS,
            supported_language=LANGUAGE,
            name="KrAccountRecognizer",
        ),
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
_TIE_PRIORITY = {entity: i for i, entity in enumerate(SUPPORTED_ENTITIES)}


def _remove_overlaps(results: list[RecognizerResult]) -> list[RecognizerResult]:
    """Keep the highest-scoring result when spans overlap."""
    return select_non_overlapping(results, _TIE_PRIORITY)


class KoreanPIIGuard:
    """Detect and mask Korean personal information.

    Args:
        entities: Entity types to detect. Defaults to structured identifiers.
            KR_NAME and KR_ADDRESS are opt-in, even when ner is supplied.
        score_threshold: Minimum confidence (0-1) for a span to count as PII.
            Korean context words (e.g. "주민번호", "연락처") near a match raise
            its confidence.
        normalize_unicode: Fold identifier typography while preserving original
            text and offsets. Disable to analyze only the literal input.
        ner: Optional local contextual name/address model. Without it, KR_NAME
            detects explicit fields and KR_ADDRESS uses bounded address grammar.
    """

    def __init__(
        self,
        entities: Iterable[str] | None = None,
        score_threshold: float = 0.4,
        *,
        normalize_unicode: bool = True,
        ner: NameAddressNER | None = None,
    ):
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1")
        self.entities: list[str] = list(DEFAULT_ENTITIES) if entities is None else list(entities)
        if not self.entities:
            raise ValueError("entities must not be empty; pass None for default types")
        unknown = set(self.entities) - set(SUPPORTED_ENTITIES)
        if unknown:
            raise ValueError(f"Unsupported entities: {sorted(unknown)}")
        self.score_threshold = score_threshold
        self.normalize_unicode = normalize_unicode
        self.ner = ner
        self._analyzer = _build_analyzer()

    def _raw_results(self, text: str) -> list[RecognizerResult]:
        normalized = normalize_text(text) if self.normalize_unicode else NormalizedText(text)
        text = normalized.text
        requested = [e for e in self.entities if e in _STRUCTURED_ENTITIES]
        results = collect_results(
            self._analyzer,
            text=text,
            # Even account-only detection must reject recognizable identifiers.
            entities=list(_STRUCTURED_ENTITIES) if "KR_ACCOUNT" in self.entities else requested,
        ) if requested else []
        boosted = require_context_for_undelimited(text, boost_with_korean_context(text, results))
        # A mistyped identifier explicitly labeled as personal data still needs
        # masking. Unlabeled impossible birth dates are not reliable evidence.
        boosted = [r for r in boosted if (
            has_valid_registration_date(r.entity_type, text[r.start:r.end])
            or has_korean_context(text, r.start, r.entity_type)
        )]
        protected = [r for r in boosted if r.entity_type != "KR_ACCOUNT" and r.score >= 0.4]
        protected_index = SpanIndex(protected)
        boosted = [
            r for r in boosted
            if r.entity_type != "KR_ACCOUNT"
            or not protected_index.overlaps(r.start, r.end)
        ]
        # ISO dates fit the broad three-group account pattern.
        boosted = [
            r for r in boosted
            if r.entity_type != "KR_ACCOUNT"
            or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text[r.start:r.end])
        ]
        filtered = [
            r for r in boosted
            if r.entity_type in self.entities and r.score >= self.score_threshold
        ]
        selected = _remove_overlaps(filtered)
        extra = []
        if "KR_NAME" in self.entities:
            extra.extend(recognize_name_fields(text))
        if "KR_ADDRESS" in self.entities:
            extra.extend(recognize_addresses(text))
        if self.ner is not None and {"KR_NAME", "KR_ADDRESS"}.intersection(self.entities):
            model_results = self.ner.analyze(text)
            for result in model_results:
                if (result.entity_type not in ("KR_NAME", "KR_ADDRESS")
                        or type(result.start) is not int
                        or type(result.end) is not int
                        or not 0 <= result.start < result.end <= len(text)
                        or not 0 <= result.score <= 1):
                    raise ValueError("NER returned an invalid name/address span")
            # Explicit fields/grammatical addresses take priority over a model's
            # shorter or differently segmented interpretation of the same value.
            rule_spans = SpanIndex([r for r in extra if r.score >= self.score_threshold])
            extra.extend(r for r in model_results
                         if not rule_spans.overlaps(r.start, r.end)
                         and (allows_contextual_name(text, r.start, r.end)
                              if r.entity_type == "KR_NAME"
                              else any(c.isdecimal() for c in text[r.start:r.end])))
        # A name-looking road token must not split a complete address. Existing
        # structured identifiers also remain authoritative in overlapping spans.
        structured_spans = SpanIndex(selected)
        address_spans = SpanIndex([r for r in extra if r.entity_type == "KR_ADDRESS"
                                   and r.entity_type in self.entities
                                   and r.score >= self.score_threshold
                                   and not structured_spans.overlaps(r.start, r.end)])
        extra = [r for r in extra if r.entity_type in self.entities
                 and r.score >= self.score_threshold
                 and not structured_spans.overlaps(r.start, r.end)
                 and (r.entity_type != "KR_NAME" or not address_spans.overlaps(r.start, r.end))]
        selected = sorted([*selected, *_remove_overlaps(extra)], key=lambda r: r.start)
        return [RecognizerResult(
            entity_type=r.entity_type,
            start=normalized.original_span(r.start, r.end)[0],
            end=normalized.original_span(r.start, r.end)[1], score=r.score,
            recognition_metadata=r.recognition_metadata,
        ) for r in selected]

    def analyze(self, text: str) -> list[Finding]:
        """Return PII spans found in ``text``, ordered by position."""
        return [
            Finding(r.entity_type, r.start, r.end, round(r.score, 3), text[r.start : r.end])
            for r in self._raw_results(text)
        ]

    def contains_pii(self, text: str) -> bool:
        """Return True if ``text`` contains any PII above the threshold."""
        return bool(self._raw_results(text))

    def mask(self, text: str, style: MaskStyle = "tag", *,
             should_mask: Callable[[Finding], bool] | None = None) -> str:
        """Return ``text`` with PII replaced.

        ``should_mask`` optionally selects findings for replacement. Returning
        False preserves that span verbatim, for caller-owned policies such as
        keeping a filename usable. It does not change analyze/contains_pii.

        Styles:
            ``tag``: ``<KR_RRN>``
            ``stars``: every character replaced with ``*``
            ``partial``: keep a short prefix: 6 characters for RRN/FRN
                (``900101-*******``), 3 for phone numbers (``010-****-****``),
                3 for account numbers, 1 for names, 0 for addresses,
                and 2 for everything else
        """
        if style not in ("tag", "stars", "partial"):
            raise ValueError(f"Unknown mask style: {style}")
        results = self._raw_results(text)
        if should_mask is not None:
            results = [r for r in results if should_mask(Finding(
                r.entity_type, r.start, r.end, round(r.score, 3), text[r.start:r.end]
            ))]
        if not results:
            return text
        if style == "partial":
            return self._partial_mask(text, results)
        out: list[str] = []
        cursor = 0
        for result in results:
            out.append(text[cursor:result.start])
            out.append(f"<{result.entity_type}>" if style == "tag"
                       else "*" * (result.end - result.start))
            cursor = result.end
        out.append(text[cursor:])
        return "".join(out)

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
