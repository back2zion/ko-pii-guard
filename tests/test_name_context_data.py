"""Validate authored contrasts and exact gold spans without running a recognizer."""

import hashlib
import json
import runpy
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "benchmarks/name_context_cases.py"
DATA = ROOT / "benchmarks/data/name_context.jsonl"
API = runpy.run_path(str(GENERATOR))
CASES = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()]
SPLITS = ("calibration", "challenge", "seen_surface_challenge")


def _rows(split, track=None):
    return [
        case for case in CASES
        if case["split"] == split and (track is None or case["track"] == track)
    ]


def test_frozen_corpus_is_reproducible_without_model_predictions():
    first = API["build_cases"]()
    assert first == API["build_cases"]() == CASES
    assert API["serialize_cases"](first) == DATA.read_text(encoding="utf-8")


def test_cases_have_unique_identity_and_exact_authored_name_spans():
    required = {
        "id", "group_id", "context_id", "surface_id", "surface", "name_family",
        "split", "track", "text", "expected",
    }
    assert len({case["id"] for case in CASES}) == len(CASES)
    assert len({case["text"] for case in CASES}) == len(CASES)
    for case in CASES:
        assert required <= case.keys()
        assert case["split"] in SPLITS
        assert case["name_family"] in API["FAMILIES"]
        assert case["surface"] in case["text"]
        assert case["track"] in {"person_context", "non_person_context", "boundary"}
        if case["track"] == "non_person_context":
            assert case["expected"] == []
        else:
            assert len(case["expected"]) == 1
            gold = case["expected"][0]
            assert gold["entity"] == "KR_NAME"
            assert type(gold["start"]) is int and type(gold["end"]) is int
            assert 0 <= gold["start"] < gold["end"] <= len(case["text"])
            assert case["text"][gold["start"]:gold["end"]] == case["surface"]


def test_original_splits_have_disjoint_target_surfaces_and_context_families():
    calibration, challenge = _rows("calibration"), _rows("challenge")
    for key in ("surface", "surface_id", "context_id", "group_id"):
        assert {case[key] for case in calibration}.isdisjoint(case[key] for case in challenge)
    templates = API["PERSON_CONTEXTS"]
    assert {template for _, template in templates["calibration"]}.isdisjoint(
        template for _, template in templates["challenge"]
    )
    # Disjoint targets are the contract: ordinary background tokens need not be
    # absent, nor are surface spellings claimed unseen in model pretraining.
    assert "하나" not in {case["surface"] for case in challenge}
    assert any("하나" in case["text"] for case in challenge)


def test_seen_surface_challenge_has_only_the_declared_intentional_overlap():
    calibration, challenge, seen = (_rows(split) for split in SPLITS)
    for key in ("surface", "surface_id"):
        assert {case[key] for case in seen} == {case[key] for case in calibration}
    assert {case["context_id"] for case in seen if case["track"] == "person_context"} == {
        case["context_id"] for case in challenge if case["track"] == "person_context"
    }
    seen_negatives = _rows("seen_surface_challenge", "non_person_context")
    assert {case["context_id"] for case in seen_negatives}.isdisjoint(
        case["context_id"] for case in calibration + challenge
    )
    assert {case["group_id"] for case in seen}.isdisjoint(
        case["group_id"] for case in calibration + challenge
    )


def test_each_person_context_is_crossed_with_all_surface_families():
    for split in SPLITS:
        rows = _rows(split, "person_context")
        expected_surfaces = {case["surface"] for case in _rows(split)}
        by_context = defaultdict(list)
        for case in rows:
            by_context[case["context_id"]].append(case)
        assert len(by_context) == (8 if split == "calibration" else 16)
        for group in by_context.values():
            assert {case["surface"] for case in group} == expected_surfaces
            assert {case["name_family"] for case in group} == set(API["FAMILIES"])


def test_homonymous_words_have_both_person_and_ordinary_word_contrasts():
    for split in SPLITS:
        expected_words = set(API["NON_PERSON_CONTEXTS"][split])
        homonyms = [case for case in _rows(split) if case["name_family"] == "homonymous_word"]
        assert {case["surface"] for case in homonyms} == expected_words
        for surface in expected_words:
            group = [case for case in homonyms if case["surface"] == surface]
            assert len({case["group_id"] for case in group}) == 1
            assert {case["track"] for case in group} >= {"person_context", "non_person_context"}
            negatives = [case for case in group if case["track"] == "non_person_context"]
            assert len(negatives) == (25 if split == "challenge" else 12)
        assert all(
            case["name_family"] == "homonymous_word"
            for case in _rows(split, "non_person_context")
        )


def test_inherent_name_endings_remain_in_gold_and_particles_stay_out():
    expected_boundaries = (
        ("boundary_possessive_photo", "이상은", "이상은의", "의"),
        ("boundary_possessive_photo", "이상", "이상의", "의"),
        ("boundary_topic_key", "오지은", "오지은은", "은"),
        ("boundary_subject_umbrella", "김하나", "김하나가", "가"),
    )
    for context_id, surface, phrase, particle in expected_boundaries:
        matches = [
            case for case in _rows("challenge", "boundary")
            if case["context_id"] == context_id and case["surface"] == surface
        ]
        assert len(matches) == 1
        case = matches[0]
        gold = case["expected"][0]
        assert phrase in case["text"]
        assert case["text"][gold["start"]:gold["end"]] == surface
        assert case["text"][gold["end"]] == particle


def test_authored_quotatives_agree_with_name_endings():
    rows = {case["surface"]: case for case in _rows("calibration")
            if case["context_id"] == "signed_application"}
    for name, phrase in (("김서윤", "김서윤이라고"), ("최지우", "최지우라고"),
                         ("하나", "하나라고"), ("미소", "미소라고")):
        assert phrase in rows[name]["text"]


def test_metadata_reports_dependence_unique_contexts_and_fixed_data_hash():
    metadata = API["corpus_metadata"](CASES)
    assert metadata["sentences"] == 808
    assert metadata["positive_sentences"] == 636
    assert metadata["negative_sentences"] == 172
    assert metadata["unique_surfaces"] == 32
    assert metadata["unique_contexts"] == 203
    assert metadata["surface_groups"] == 44
    assert metadata["paired_positive_negative_groups"] == 10
    assert {
        split: metadata["per_split"][split]["paired_positive_negative_groups"] for split in SPLITS
    } == {"calibration": 3, "challenge": 4, "seen_surface_challenge": 3}
    assert {split: metadata["per_split"][split]["sentences"] for split in SPLITS} == {
        "calibration": 132, "challenge": 448, "seen_surface_challenge": 228,
    }
    assert Counter(case["split"] for case in CASES) == {
        "calibration": 132, "challenge": 448, "seen_surface_challenge": 228,
    }
    assert metadata["data_sha256"] == hashlib.sha256(DATA.read_bytes()).hexdigest()
    assert metadata["generator_sha256"] == hashlib.sha256(GENERATOR.read_bytes()).hexdigest()
    assert metadata["pretraining_exposure"] == "unknown"
    assert metadata["nrb_reproduction"] is False
    assert metadata["population_rate_estimate"] is False
    assert "not independent" in metadata["independence_warning"]
