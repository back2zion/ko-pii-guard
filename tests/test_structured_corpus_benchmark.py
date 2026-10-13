"""Whole-field evaluation must score masking without exporting private values."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.structured_corpus_benchmark import paired_leaves, read_cases

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "structured_corpus_benchmark.py"


def test_paired_leaves_distinguishes_null_from_masked_key():
    original = {"010-1234-5678": None}
    tagged = {"<PHONE_NUMBER>": None}
    starred = {"*" * 13: None}
    assert list(paired_leaves(original, tagged, starred)) == [
        (("010-1234-5678",), None, None, None, False),
        (("010-1234-5678",), None, None, None, True),
    ]


@pytest.mark.parametrize("invalid", [
    '{"id":"secret-id","record":{"phone":"010-1234-5678"},"targets":'
    '[{"path":["missing"],"type":"PHONE_NUMBER"}]}',
    '{"id":"secret-id","record":{"phone":"010-1234-5678"},"targets":'
    '[{"path":["phone"],"type":"PHONE_NUMBER"}],"negative_paths":[["phone"]]}',
    '{"id":"secret-id","record":{"phone":"010-1234-5678",'
    '"phone":"other"},"targets":[]}',
    '{"id":"secret-id","record":{"amount":1e999},"targets":[]}',
])
def test_invalid_annotations_do_not_echo_values(tmp_path, invalid):
    path = tmp_path / "private.jsonl"
    path.write_text(invalid + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1") as error:
        read_cases(path, {"PHONE_NUMBER"})
    assert "secret-id" not in str(error.value)
    assert "010-1234-5678" not in str(error.value)


def test_validation_and_comparison_report_aggregate_only(tmp_path):
    secret = "010-1234-5678"
    path, output = tmp_path / "private.jsonl", tmp_path / "report.json"
    path.write_text(json.dumps({
        "id": "secret-id", "record": {"phone": secret, "order": "12345"},
        "targets": [{"path": ["phone"], "type": "PHONE_NUMBER"}],
        "negative_paths": [["order"]],
    }, ensure_ascii=False) + "\n" + json.dumps({
        "id": "clean-id", "record": {"note": "접수", "nothing": None},
        "targets": [],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(SCRIPT), "--input", str(path),
                    "--output", str(output), "--validate-only"],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    validation = json.loads(output.read_text())
    assert validation["target_fields"] == 1
    assert validation["negative_records"] == 1
    subprocess.run([sys.executable, str(SCRIPT), "--input", str(path),
                    "--output", str(output), "--baseline-ref", "HEAD"],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    report = json.loads(output.read_text())
    assert report["current"]["fully_masked_fields"] == 1
    assert report["current"]["exact_type_fields"] == 1
    assert report["current"]["changed_hard_negative_fields"] == 0
    assert report["current"]["negative_records_with_changed_field"] == 0
    assert report["baseline"]["target_fields"] == 1
    assert report["scorer_sha256"]
    assert secret not in output.read_text()
    assert "secret-id" not in output.read_text()
    assert output.stat().st_mode & 0o777 == 0o600
