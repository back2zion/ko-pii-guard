"""Public API regression contracts for one-pass screening and output offsets."""

import pytest
from presidio_analyzer import RecognizerResult

from ko_pii_guard import KoreanPIIGuard


@pytest.mark.parametrize("operation", ["analyze", "contains_pii", "mask"])
def test_generator_ner_does_not_silently_lose_findings(operation):
    class GeneratorNER:
        def analyze(self, text):
            yield RecognizerResult("KR_NAME", 0, 2, .99)

    guard = KoreanPIIGuard(entities=["KR_NAME"], ner=GeneratorNER())
    result = getattr(guard, operation)("이상 씨가 방문했습니다.")
    if operation == "analyze":
        assert [finding.text for finding in result] == ["이상"]
    elif operation == "contains_pii":
        assert result is True
    else:
        assert result == "<KR_NAME> 씨가 방문했습니다."


@pytest.mark.parametrize("operation", ["analyze", "contains_pii", "mask"])
def test_noniterable_ner_output_fails_explicitly(operation):
    class InvalidNER:
        def analyze(self, text):
            return None

    guard = KoreanPIIGuard(entities=["KR_NAME"], ner=InvalidNER())
    with pytest.raises(ValueError, match="NER returned"):
        getattr(guard, operation)("이상 씨가 방문했습니다.")


@pytest.mark.parametrize("operation", ["analyze", "contains_pii", "mask"])
@pytest.mark.parametrize("value", [None, 123, b"010-1234-5678", ["test"]])
def test_text_input_has_consistent_validation(operation, value):
    with pytest.raises(TypeError, match="text must be a string"):
        getattr(KoreanPIIGuard(), operation)(value)


@pytest.mark.parametrize("threshold", [True, False, None, "0.4"])
def test_threshold_rejects_nonnumeric_and_boolean_values(threshold):
    with pytest.raises(ValueError, match="score_threshold"):
        KoreanPIIGuard(score_threshold=threshold)


def test_entities_string_is_not_treated_as_characters():
    with pytest.raises(TypeError, match="entities"):
        KoreanPIIGuard(entities="KR_RRN")


def test_guardrails_fix_scans_custom_ner_once():
    pytest.importorskip("guardrails")
    from ko_pii_guard.guardrails_validator import KoreanPII

    class CountingNER:
        calls = 0

        def analyze(self, text):
            self.calls += 1
            return [RecognizerResult("KR_NAME", 0, 2, .99)]

    backend = CountingNER()
    validator = KoreanPII(entities=["KR_NAME"], on_fail="fix")
    validator._guard.ner = backend
    result = validator.validate("이상 씨가 방문했습니다.")
    assert result.fix_value == "<KR_NAME> 씨가 방문했습니다."
    assert backend.calls == 1
