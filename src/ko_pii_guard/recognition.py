"""Run Presidio's public pattern validators without quadratic deduplication.

This runner is deliberately limited to the recognizers registered by this
package. Korean context and overlap policies are applied by KoreanPIIGuard.
It does not implement AnalyzerEngine's generic NLP, allow-list or tracing API.
"""

from __future__ import annotations

import phonenumbers
import regex
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerResult
from presidio_analyzer.predefined_recognizers import PhoneRecognizer

from ko_pii_guard.spans import deduplicate_results


def collect_results(
    analyzer: AnalyzerEngine, text: str, entities: list[str]
) -> list[RecognizerResult]:
    results: list[RecognizerResult] = []
    for recognizer in analyzer.registry.get_recognizers(language="ko", entities=entities):
        if not recognizer.is_loaded:
            recognizer.load()
            recognizer.is_loaded = True
        if isinstance(recognizer, PhoneRecognizer):
            for region in recognizer.supported_regions:
                for match in phonenumbers.PhoneNumberMatcher(
                    text, region, leniency=recognizer.leniency
                ):
                    results.append(RecognizerResult(
                        entity_type=recognizer.supported_entities[0],
                        start=match.start, end=match.end, score=recognizer.SCORE,
                        recognition_metadata={
                            RecognizerResult.RECOGNIZER_NAME_KEY: recognizer.name,
                            RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: recognizer.id,
                        },
                    ))
            continue
        if not isinstance(recognizer, PatternRecognizer):
            results.extend(recognizer.analyze(text, entities, nlp_artifacts=None))
            continue
        for pattern in recognizer.patterns:
            if pattern.compiled_regex is None or (
                pattern.compiled_with_flags != recognizer.global_regex_flags
            ):
                pattern.compiled_regex = regex.compile(
                    pattern.regex, flags=recognizer.global_regex_flags
                )
                pattern.compiled_with_flags = recognizer.global_regex_flags
            # A timeout must abort analysis rather than return partially masked
            # text as if it were safe. No PII text is included in the exception.
            for match in pattern.compiled_regex.finditer(text, timeout=1.0):
                start, end = match.span()
                if start == end:
                    continue
                value = match.group()
                valid = recognizer.validate_result(value)
                invalid = recognizer.invalidate_result(value)
                score = pattern.score if valid is None else (1.0 if valid else 0.0)
                if invalid or score <= 0:
                    continue
                results.append(RecognizerResult(
                    entity_type=recognizer.supported_entities[0], start=start, end=end,
                    score=score,
                    recognition_metadata={
                        RecognizerResult.RECOGNIZER_NAME_KEY: recognizer.name,
                        RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: recognizer.id,
                    },
                ))
    return deduplicate_results(results)
