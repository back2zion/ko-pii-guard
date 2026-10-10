"""Frozen cache collection rejects incomplete models and unconsumed sources."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("presidio_analyzer")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from collect_name_frozen_development import (
        digest,
        validate_cases,
        validate_checkpoint,
        validate_consumed_source,
    )
finally:
    sys.path[:] = previous_path


def write_json(path, value):
    path.write_text(json.dumps(value))


def checkpoint(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    training_source = tmp_path / "training-source.py"
    training_source.write_text("original training source")
    write_json(directory / "manifest.json", {
        "epochs": 2, "model_id": "fixed", "revision": "rev",
        "source_sha256": {str(training_source.resolve()): digest(training_source)},
    })
    write_json(directory / "config.json", {
        "selected_epoch": 1, "threshold": 0.5, "model_id": "fixed", "revision": "rev",
    })
    write_json(directory / "report.json", {
        "source_changed": False, "selected_epoch": 1, "threshold": 0.5,
        "history": [{"epoch": 1}, {"epoch": 2}],
    })
    (directory / "head.safetensors").write_bytes(b"fixture weights")
    (directory / "trainer.py.source.txt").write_text("archived executed source")
    return directory, training_source


def test_completed_checkpoint_pins_every_artifact_and_validates_original_training_source(tmp_path):
    directory, _ = checkpoint(tmp_path)
    hashes = validate_checkpoint(directory)
    assert set(hashes) == {str(p.resolve()) for p in directory.iterdir()}
    assert hashes[str((directory / "trainer.py.source.txt").resolve())] == digest(
        directory / "trainer.py.source.txt")


@pytest.mark.parametrize("mutation", ["incomplete", "selection", "source_flag", "source_file"])
def test_incomplete_or_inconsistent_checkpoint_is_rejected(tmp_path, mutation):
    directory, source = checkpoint(tmp_path)
    report = json.loads((directory / "report.json").read_text())
    if mutation == "incomplete":
        report["history"] = report["history"][:1]
    elif mutation == "selection":
        report["threshold"] = 0.75
    elif mutation == "source_flag":
        report["source_changed"] = True
    else:
        source.write_text("changed training source")
    write_json(directory / "report.json", report)
    with pytest.raises(ValueError):
        validate_checkpoint(directory)


def test_consumed_report_requires_completed_scope_unique_ids_and_pinned_source(tmp_path):
    source = tmp_path / "consumed.jsonl"
    source.write_text("fixed source bytes")
    report = {"scope": "development", "source_changed_during_run": False,
              "candidate": {"metrics": {"tp": 1}}, "baseline": {"metrics": {"tp": 1}},
              "selected_ids": ["old-id"],
              "frozen_sha256": {str(source.resolve()): digest(source)}}
    assert validate_consumed_source(report, "klue", source) == report["frozen_sha256"]
    for change in ({"scope": "heldout"}, {"selected_ids": []},
                   {"selected_ids": ["old-id", "old-id"]},
                   {"source_changed_during_run": True}):
        with pytest.raises(ValueError):
            validate_consumed_source(dict(report, **change), "klue", source)
    source.write_text("different text under the same ID")
    with pytest.raises(ValueError):
        validate_consumed_source(report, "klue", source)


def test_cases_retain_duplicate_text_but_reject_ids_or_normalization_changes():
    cases = [{"id": "one", "text": "홍길동"}, {"id": "two", "text": "홍길동"}]
    validate_cases(cases, ["one", "two"])
    with pytest.raises(ValueError):
        validate_cases(cases[::-1], ["one", "two"])
    with pytest.raises(ValueError):
        validate_cases([{"id": "one", "text": "Ａ씨"}], ["one"])
    with pytest.raises(ValueError):
        validate_cases([cases[0], cases[0]], ["one", "one"])
