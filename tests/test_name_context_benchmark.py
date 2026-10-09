"""Accounting and score-ablation tests; injected outputs are not model evidence."""

import sys
from pathlib import Path

import pytest
from presidio_analyzer import RecognizerResult

from ko_pii_guard import KoreanPIIGuard

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import name_context_benchmark as benchmark  # noqa: E402


class FakeRawModel:
    def __init__(self, predictions):
        self.predictions = predictions
        self.calls = []

    def analyze(self, text):
        self.calls.append(text)
        return self.predictions[text]


def result(text, value, *, entity="KR_NAME", score=0.97):
    start = text.index(value)
    return RecognizerResult(entity, start, start + len(value), score)


def case(case_id, text, *names, surface="이상", split="challenge", context="colleague"):
    return {
        "id": case_id, "text": text, "surface": surface, "split": split,
        "track": "person_context" if names else "nonperson_context",
        "context_id": context, "group_id": surface, "name_family": "homonym",
        "expected": [{"entity": "KR_NAME", "start": text.index(name),
                      "end": text.index(name) + len(name)} for name in names],
    }


def guard_for(predictions, lexicon=frozenset(), boost=0.0):
    model = FakeRawModel(predictions)
    cache = benchmark.RawCache(model)
    backend = benchmark.ScoreOnlyNER(cache, lexicon, boost)
    return KoreanPIIGuard(entities=["KR_NAME"], ner=backend), cache, model


def test_comparisons_share_one_raw_call_and_cannot_mutate_cached_predictions():
    text = "동료 소망이 서류를 제출했습니다."
    original = result(text, "소망", score=0.87)
    model = FakeRawModel({text: [original]})
    cache = benchmark.RawCache(model)
    baseline = benchmark.ScoreOnlyNER(cache, frozenset(), 0.0)
    boosted = benchmark.ScoreOnlyNER(cache, frozenset({"소망"}), benchmark.BOOST)
    assert baseline.analyze(text) == []
    snapshot = cache.get(text)
    [candidate] = boosted.analyze(text)
    assert candidate.score == pytest.approx(0.92)

    candidate.start, candidate.score = 0, 0.0
    original.end, original.score = len(text), 1.0
    [fresh] = boosted.analyze(text)
    assert (fresh.start, fresh.end) == (3, 5)
    assert fresh.score == pytest.approx(0.92)
    assert fresh is not candidate
    assert cache.get(text) == snapshot == (("KR_NAME", 3, 5, 0.87),)
    assert baseline.scored(text) == list(snapshot)
    assert model.calls == [text]


@pytest.mark.parametrize("entity,value,score,expected_score,accepted", [
    ("KR_NAME", "소망", 0.85, 0.90, True),
    ("KR_NAME", "소망", 0.849, 0.899, False),
    ("KR_NAME", "소망", 0.97, 1.0, True),
    ("KR_NAME", "소망은", 0.87, 0.87, False),
    ("KR_NAME", "망", 0.87, 0.87, False),
    ("KR_NAME", "김하나", 0.87, 0.87, False),
    ("KR_NAME", "김하나", 0.93, 0.93, True),
    ("KR_ADDRESS", "소망", 0.87, 0.87, False),
    ("KR_ADDRESS", "소망", 0.93, 0.93, True),
])
def test_score_boost_changes_only_exact_name_surface_and_preserves_candidate_boundaries(
    entity, value, score, expected_score, accepted
):
    text = f"앞 {value} 뒤"
    cache = benchmark.RawCache(FakeRawModel({text: [result(text, value, entity=entity,
                                                        score=score)]}))
    backend = benchmark.ScoreOnlyNER(cache, frozenset({"소망"}), benchmark.BOOST)
    [scored] = backend.scored(text)
    assert scored[:3] == cache.get(text)[0][:3] == (entity, 2, 2 + len(value))
    assert scored[3] == pytest.approx(expected_score)
    assert bool(backend.analyze(text)) is accepted
    assert cache.get(text)[0][3] == score


def test_lexicon_does_not_create_candidates_when_model_finds_none():
    text = "동료 소망이 서류를 제출했습니다."
    guard, cache, model = guard_for({text: []}, frozenset({"소망"}), benchmark.BOOST)
    assert guard.analyze(text) == []
    assert guard.mask(text) == text
    assert cache.get(text) == ()
    assert model.calls == [text]


def test_lexicon_uses_only_calibration_name_positives():
    rows = [
        case("cal-positive", "동료 이상이 왔습니다.", "이상", split="calibration"),
        case("cal-duplicate", "이상 씨가 왔습니다.", "이상", split="calibration"),
        case("cal-negative", "소망을 적으세요.", surface="소망", split="calibration"),
        case("unseen-positive", "김하나 씨가 왔습니다.", "김하나", surface="김하나"),
        case("heldout-positive", "박이 씨가 왔습니다.", "박이", surface="박이",
             split="heldout_context"),
        {"split": "calibration", "surface": "서울", "expected": [
            {"entity": "KR_ADDRESS", "start": 0, "end": 2},
        ]},
    ]
    assert benchmark.calibration_lexicon(rows) == frozenset({"이상"})


def test_score_promotion_can_create_a_homonym_false_positive_without_new_raw_candidates():
    text = "이상 없습니다."
    row = case("homonym-negative", text)
    model = FakeRawModel({text: [result(text, "이상", score=0.87)]})
    cache = benchmark.RawCache(model)
    baseline = KoreanPIIGuard(
        entities=["KR_NAME"], ner=benchmark.ScoreOnlyNER(cache, frozenset(), 0.0)
    )
    boosted = KoreanPIIGuard(
        entities=["KR_NAME"],
        ner=benchmark.ScoreOnlyNER(cache, frozenset({"이상"}), benchmark.BOOST),
    )
    before, after = benchmark.observe(row, baseline), benchmark.observe(row, boosted)
    assert before["correct"] is True
    assert before["counts"]["masked_negative_characters"] == 0
    assert after["correct"] is False
    assert after["predicted"] == [("KR_NAME", 0, 2)]
    assert after["counts"]["false_positive"] == 1
    assert after["counts"]["false_positive_sentences"] == 1
    assert after["counts"]["masked_negative_characters"] == 2
    assert after["counts"]["negative_characters"] == len(text)
    assert boosted.mask(text, style="stars") == "** 없습니다."
    assert benchmark.metrics([after])["negative_masked_character_ratio"] == 2 / len(text)
    assert model.calls == [text]
    assert cache.get(text) == (("KR_NAME", 0, 2, 0.87),)


@pytest.mark.parametrize("predictions,expected", [
    (["이상은"], (1, 0, 0, 1, 0, 0, 0, 0, 0)),
    (["이상은의"], (0, 1, 1, 1, 0, 1, 0, 1, 0)),
    (["이상"], (0, 1, 1, 0, 1, 0, 0, 1, 0)),
    ([], (0, 0, 1, 0, 0, 0, 1, 0, 0)),
    (["이상", "은"], (0, 2, 1, 1, 0, 0, 0, 1, 1)),
])
def test_exact_name_boundaries_and_complete_masking_are_independent(predictions, expected):
    text = "동료 이상은의 의견입니다."
    row = case("name-boundary", text, "이상은", surface="이상은")
    guard, _, _ = guard_for({text: [result(text, value) for value in predictions]})
    observation = benchmark.observe(row, guard)
    fields = ("true_positive", "false_positive", "false_negative", "fully_masked_gold",
              "partial_gold", "overwide_gold", "missed_gold", "boundary_error_gold", "split_gold")
    assert tuple(observation["counts"][field] for field in fields) == expected
    assert observation["correct"] is (predictions == ["이상은"])
    assert observation["counts"]["masked_non_gold_characters"] == (
        1 if predictions == ["이상은의"] else 0
    )


def test_merged_names_mask_fully_but_count_as_two_boundary_errors():
    text = "동료 이상은과 김하나가 왔습니다."
    row = case("merged-names", text, "이상은", "김하나", surface="이상은")
    guard, _, _ = guard_for({text: [result(text, "이상은과 김하나")]})
    observation = benchmark.observe(row, guard)
    counts = observation["counts"]
    assert observation["correct"] is False
    assert (counts["true_positive"], counts["false_positive"], counts["false_negative"]) == (
        0, 1, 2,
    )
    assert counts["fully_masked_gold"] == counts["boundary_error_gold"] == 2
    assert counts["overwide_gold"] == counts["masked_non_gold_characters"] == 2
    assert counts["partial_gold"] == counts["missed_gold"] == counts["split_gold"] == 0


@pytest.mark.parametrize("bad_ids,correct_groups,correct_frames", [
    ([], 2, 1),
    (["이상-negative"], 1, 1),
    (["이상-positive"], 1, 0),
    (["이상-negative", "소망-negative"], 0, 1),
])
def test_paired_success_requires_every_person_and_nonperson_sentence_to_be_exact(
    bad_ids, correct_groups, correct_frames
):
    rows = [
        case("이상-positive", "동료 이상이 왔습니다.", "이상"),
        case("이상-negative", "이상 없습니다.", context="ordinary"),
        case("소망-positive", "동료 소망이 왔습니다.", "소망", surface="소망"),
        case("소망-negative", "새해 소망을 적으세요.", surface="소망", context="ordinary"),
    ]
    predictions = {}
    for row in rows:
        values = []
        if row["expected"] or row["id"] in bad_ids:
            candidate = result(row["text"], row["surface"])
            if row["expected"] and row["id"] in bad_ids:
                candidate.end -= 1  # Detection alone must not pass an exact-span pair.
            values.append(candidate)
        predictions[row["text"]] = values
    guard, _, _ = guard_for(predictions)
    observations = {row["id"]: benchmark.observe(row, guard) for row in rows}
    paired = benchmark.paired_metrics(rows, observations)
    assert paired["person_nonperson_surface_groups"] == 2
    assert paired["all_correct_surface_groups"] == correct_groups
    assert paired["person_context_frames"] == 1
    assert paired["all_correct_frames_across_surfaces"] == correct_frames


def test_metrics_aggregate_span_counts_and_keep_undefined_positive_rates_null():
    positive = "동료 이상은의 의견입니다."
    negative = "이상 없습니다."
    correct_guard, _, _ = guard_for({positive: [result(positive, "이상은")]})
    wrong_guard, _, _ = guard_for({positive: [result(positive, "이상은의")],
                                 negative: [result(negative, "이상")]})
    row = case("positive", positive, "이상은", surface="이상은")
    observations = [benchmark.observe(row, correct_guard), benchmark.observe(row, wrong_guard),
                    benchmark.observe(case("negative", negative), wrong_guard)]
    measured = benchmark.metrics(observations)
    assert measured["exact_span_precision"] == 1 / 3
    assert measured["exact_span_recall"] == 1 / 2
    assert measured["exact_span_f1"] == 2 / 5
    assert measured["full_mask_coverage"] == 1.0
    assert measured["negative_masked_character_ratio"] == 2 / len(negative)

    empty_guard, _, _ = guard_for({negative: []})
    negative_only = benchmark.metrics([benchmark.observe(case("negative", negative), empty_guard)])
    for key in ("exact_span_precision", "exact_span_recall", "exact_span_f1", "full_mask_coverage"):
        assert negative_only[key] is None
    assert negative_only["negative_masked_character_ratio"] == 0.0
