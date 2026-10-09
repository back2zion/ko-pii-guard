import random

import pytest
import synthetic as s
from presidio_analyzer import Pattern, PatternRecognizer

from ko_pii_guard.analyzer import DEFAULT_ENTITIES, _build_analyzer
from ko_pii_guard.recognition import collect_results


def _spans(results):
    return sorted((r.entity_type, r.start, r.end, r.score) for r in results)


def test_optimized_runner_matches_presidio_validators():
    analyzer = _build_analyzer()
    rng = random.Random(31013)
    for _ in range(40):
        text = " | ".join((s.rrn(rng), s.brn(rng), s.mobile(rng), s.email(rng),
                           s.card(rng), s.passport(rng), s.account(rng)))
        expected = analyzer.analyze(text, "ko", list(DEFAULT_ENTITIES), score_threshold=0)
        assert _spans(collect_results(analyzer, text, list(DEFAULT_ENTITIES))) == _spans(expected)


def test_timeout_aborts_instead_of_returning_partial_results():
    class TimeoutPattern:
        def finditer(self, *args, **kwargs):
            raise TimeoutError("pattern timeout")

    analyzer = _build_analyzer()
    recognizer = PatternRecognizer(
        supported_entity="KR_ACCOUNT", supported_language="ko",
        patterns=[Pattern("timeout", "x", 0.4)],
    )
    pattern = recognizer.patterns[0]
    pattern.compiled_regex = TimeoutPattern()
    pattern.compiled_with_flags = recognizer.global_regex_flags
    analyzer.registry.add_recognizer(recognizer)
    with pytest.raises(TimeoutError):
        collect_results(analyzer, "계좌번호 3333-01-1234567", ["KR_ACCOUNT"])
