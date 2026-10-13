"""Compare full-field recall against typed negative overmasking."""

from benchmarks.strict_field_diagnostic import StrictGuard
from benchmarks.structured_corpus_benchmark import evaluate
from ko_pii_guard import KoreanPIIGuard


def test_strict_mode_exposes_recall_and_typed_negative_tradeoff():
    cases = [{"record": {"person_name": "김**", "phone_no": "없음"},
              "gold": {("person_name",): "KR_NAME"},
              "negative": {("phone_no",)}, "ignore": set()}]
    guard = KoreanPIIGuard(entities=["KR_NAME", "PHONE_NUMBER"])
    ordinary = evaluate(cases, guard)
    strict = evaluate(cases, StrictGuard(guard))
    assert ordinary["fully_masked_fields"] == 0
    assert ordinary["changed_hard_negative_fields"] == 0
    assert strict["fully_masked_fields"] == 1
    assert strict["changed_hard_negative_fields"] == 1
