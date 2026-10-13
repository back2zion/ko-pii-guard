"""Local evaluation must validate gold offsets and keep values out of reports."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.private_corpus_benchmark import (
    check_split_overlap,
    covered_by_predictions,
    freeze_input,
    read_cases,
)

ROOT = Path(__file__).resolve().parents[1]


def test_rejects_invalid_gold_without_echoing_text(tmp_path):
    path = tmp_path / "input.jsonl"
    secret = "010-1234-5678"
    path.write_text(json.dumps({
        "id": "one", "text": secret,
        "entities": [{"type": "PHONE_NUMBER", "start": 0, "end": 99}],
    }) + "\n")
    with pytest.raises(ValueError, match="line 1") as error:
        read_cases(path, {"PHONE_NUMBER"})
    assert secret not in str(error.value)


def test_unique_surface_derives_offsets_but_repeated_surface_needs_offsets(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps({
        "id": "one", "text": "연락처 010-1234-5678",
        "entities": [{"type": "PHONE_NUMBER", "surface": "010-1234-5678"}],
    }, ensure_ascii=False) + "\n")
    assert read_cases(path, {"PHONE_NUMBER"})[0]["gold"] == {
        ("PHONE_NUMBER", 4, 17)
    }
    path.write_text(json.dumps({
        "id": "one", "text": "010-1234-5678 / 010-1234-5678",
        "entities": [{"type": "PHONE_NUMBER", "surface": "010-1234-5678"}],
    }) + "\n")
    with pytest.raises(ValueError, match="repeated"):
        read_cases(path, {"PHONE_NUMBER"})


def test_mask_coverage_needs_a_detection_even_for_existing_stars():
    assert not covered_by_predictions(0, 4, set())
    assert not covered_by_predictions(0, 4, {("PHONE_NUMBER", 0, 2)})
    assert covered_by_predictions(0, 4, {("PHONE_NUMBER", 0, 4)})


def test_rejects_normalized_document_reuse_between_splits():
    holdout = [{"id": "h", "text": "전화 ０１０-1234-5678"}]
    development = [{"id": "d", "text": "전화 010-1234-5678"}]
    with pytest.raises(ValueError, match="share an ID or normalized document"):
        check_split_overlap(holdout, development)


def test_frozen_input_is_private_and_stable_after_source_changes(tmp_path):
    source, frozen = tmp_path / "source.jsonl", tmp_path / "frozen.jsonl"
    source.write_text("private original", encoding="utf-8")
    digest = freeze_input(source, frozen)
    source.write_text("different later", encoding="utf-8")
    assert frozen.read_text(encoding="utf-8") == "private original"
    assert digest == hashlib.sha256(b"private original").hexdigest()
    assert frozen.stat().st_mode & 0o777 == 0o600


def test_report_path_cannot_overwrite_development_input(tmp_path):
    development = tmp_path / "development.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    for path, identifier in ((development, "dev"), (holdout, "holdout")):
        path.write_text(json.dumps({"id": identifier, "text": "내용", "entities": []}) + "\n")
    original = development.read_bytes()
    result = subprocess.run([
        sys.executable, str(ROOT / "benchmarks" / "private_corpus_benchmark.py"),
        "--input", str(holdout), "--development-input", str(development),
        "--output", str(development),
    ], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert development.read_bytes() == original


def test_report_contains_aggregates_without_document_values(tmp_path):
    secret = "010-1234-5678"
    corpus = tmp_path / "input.jsonl"
    output = tmp_path / "report.json"
    positive = json.dumps({
        "id": "private-id", "text": "연락처 " + secret,
        "entities": [{"type": "PHONE_NUMBER", "start": 4, "end": 17}],
    }, ensure_ascii=False)
    negative = json.dumps({"id": "clean-id", "text": "요청 접수", "entities": []},
                          ensure_ascii=False)
    corpus.write_text(positive + "\n" + negative + "\n")
    subprocess.run([
        sys.executable, str(ROOT / "benchmarks" / "private_corpus_benchmark.py"),
        "--input", str(corpus), "--output", str(output), "--repeats", "1",
    ], cwd=ROOT, check=True, capture_output=True, text=True)
    report = json.loads(output.read_text())
    assert report["baseline"]["total"]["tp"] == 1
    assert report["current"]["full_mask_coverage"] == 1.0
    assert report["current"]["negative_document_false_positive_rate"] == 0.0
    assert report["current"]["mask_coverage_by_entity"]["PHONE_NUMBER"]["fully_masked"] == 1
    assert report["current"]["wrong_type_same_span"] == []
    assert report["dependencies"]["presidio-analyzer"]
    assert report["score_threshold"] == 0.4
    assert report["normalize_unicode"] is True
    assert report["ner_enabled"] is False
    assert report["ner_score_threshold"] is None
    assert report["ner_threads"] is None
    assert report["current"]["mask_stars_corpus_median_ms"] > 0
    assert secret not in output.read_text()
    assert "private-id" not in output.read_text()
    assert output.stat().st_mode & 0o777 == 0o600
