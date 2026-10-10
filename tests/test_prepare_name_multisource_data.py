"""Immutable original KDPII splits and cross-domain supervision safeguards."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import prepare_name_multisource_data as preparation
    from prepare_name_multisource_data import (
        FILES,
        convert_training_record,
        deduplicate,
        select_training_ids,
        verify_metadata,
        verify_source,
    )
finally:
    sys.path[:] = previous_path


def record(identifier="dialogue_1_0", text="김 별", labels=None):
    labels = labels if labels is not None else [(0, 1, "PS_NAME"), (2, 3, "PS_NICKNAME")]
    tags = ["O"] * len(text)
    entities = []
    for start, end, label in labels:
        entities.append(dict(begin=start, end=end, label=label, form=text[start:end]))
        tags[start:end] = ["B-" + label] + ["I-" + label] * (end - start - 1)
    return dict(sent_idx=identifier, sentence=text, sent_seq=list(text),
                labelling_seq=tags, PII_set=entities)


def case(identifier, text="동일 문장", split="train", expected=None, source="nsmc"):
    return dict(id=identifier, text=text, split=split, source=source,
                expected=[] if expected is None else expected)


def test_original_offsets_full_labels_and_nickname_ignore_regions_survive():
    result = convert_training_record(record(), "train")
    assert result["text"] == "김 별"
    assert result["source"] == "kdpii_v1_train"
    assert result["split"] == "train"
    assert result["expected"] == [dict(entity="KR_NAME", start=0, end=1)]
    assert result["source_annotations"][1]["label"] == "PS_NICKNAME"
    assert result["training_ignore_spans"] == [dict(start=2, end=3, label="PS_NICKNAME")]
    assert result["char_name_bio"] == ["B-PS", "O", "O"]
    assert convert_training_record(record(), "valid")["split"] == "validation"


def test_invalid_source_offsets_are_not_silently_repaired():
    bad = record()
    bad["PII_set"][0]["end"] = 2
    with pytest.raises(ValueError, match="surface"):
        convert_training_record(bad, "train")
    with pytest.raises(ValueError, match="split"):
        convert_training_record(record(), "test")


def test_training_sampling_reads_ids_only_and_is_order_independent():
    class NoLabels(dict):
        def __getitem__(self, key):
            assert key == "sent_idx", "Selection must not inspect text or labels"
            return super().__getitem__(key)
    records = [NoLabels(sent_idx=str(i)) for i in range(20)]
    assert select_training_ids(records, 7) == select_training_ids(records[::-1], 7)
    assert len(select_training_ids(records, 0)) == 20
    with pytest.raises(ValueError, match="ID"):
        select_training_ids([records[0], records[0]], 1)


def test_official_validation_wins_exact_duplicate_and_conflicts_are_quarantined():
    same = [case("t"), case("v", split="validation", source="kdpii_v1_valid")]
    kept, report = deduplicate(same)
    assert [c["id"] for c in kept] == ["v"]
    assert report["duplicate_rows_removed"] == 1
    conflicting = [*same, case("conflict", expected=[dict(entity="KR_NAME", start=0, end=2)])]
    kept, report = deduplicate(conflicting)
    assert kept == []
    assert report["conflicting_rows_removed"] == 3


def test_non_name_annotation_conflicts_within_kdpii_are_quarantined():
    one = convert_training_record(record("one", "별", [(0, 1, "PS_NICKNAME")]), "train")
    two = convert_training_record(record("two", "별", [(0, 1, "PS_ID")]), "valid")
    kept, report = deduplicate([one, two])
    assert not kept
    assert report["conflicting_rows_removed"] == 2


def test_evaluation_exclusions_use_hashes_without_reading_evaluation_data():
    row = case("excluded")
    sha = hashlib.sha256(row["text"].encode()).hexdigest()
    kept, report = deduplicate([row, case("kept", text="별도 문장")], excluded_hashes={sha})
    assert [c["id"] for c in kept] == ["kept"]
    assert report["excluded_ids"] == ["excluded"]


def test_runtime_normalized_evaluation_hash_also_excludes_original_text():
    row = case("fullwidth", text="ＡＢＣ김")
    sha = hashlib.sha256("ABC김".encode()).hexdigest()
    kept, report = deduplicate([row], excluded_hashes={sha})
    assert kept == []
    assert report["excluded_ids"] == ["fullwidth"]


def test_metadata_rejects_changed_license_or_checksum():
    metadata = dict(id=10968609, metadata=dict(license=dict(id="cc-by-4.0")),
                    files=[dict(key=name, size=spec["bytes"], checksum="md5:" + spec["md5"])
                           for name, spec in FILES.items()])
    verify_metadata(metadata)
    metadata["files"][0]["checksum"] = "md5:changed"
    with pytest.raises(ValueError, match="metadata"):
        verify_metadata(metadata)
    metadata["metadata"]["license"]["id"] = "unknown"
    with pytest.raises(ValueError, match="license"):
        verify_metadata(metadata)


def test_source_bytes_must_match_pinned_checksums_and_size():
    with pytest.raises(ValueError, match="checksum"):
        verify_source("train.json", b"[]")


def test_end_to_end_preserves_domains_excludes_both_sources_and_never_reads_test(
    tmp_path, monkeypatch,
):
    source, existing, output = (tmp_path / name for name in ("source", "existing", "output"))
    source.mkdir()
    existing.mkdir()
    raw = {
        "train.json": [record("kt1", "학습", []), record("kt2", "ＡＢＣ김", [])],
        "valid.json": [record("kv1", "검증", [])],
    }
    specs = {}
    for name, rows in raw.items():
        content = json.dumps(rows, ensure_ascii=False).encode()
        (source / name).write_bytes(content)
        specs[name] = dict(bytes=len(content), md5=hashlib.md5(content).hexdigest(),
                           sha256=hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(preparation, "FILES", specs)
    metadata = dict(id=10968609, metadata=dict(license=dict(id="cc-by-4.0"), creators=[]),
                    files=[dict(key=name, size=spec["bytes"], checksum="md5:" + spec["md5"])
                           for name, spec in specs.items()])
    (source / "record-metadata.json").write_text(json.dumps(metadata))
    (source / "test.json").write_text("TEST MUST NOT BE OPENED OR PARSED")
    prior = {
        "train.jsonl": [case("lt1", text="배제"), case("lt2", text="유지")],
        "validation.jsonl": [case("lv1", text="기존 검증", split="validation")],
        "synthetic-validation.jsonl": [],
    }
    manifest = {"outputs": {}}
    for name, rows in prior.items():
        content = "".join(json.dumps(c) + "\n" for c in rows).encode()
        (existing / name).write_bytes(content)
        manifest["outputs"][name] = dict(rows=len(rows), sha256=hashlib.sha256(content).hexdigest())
    (existing / "manifest.json").write_text(json.dumps(manifest))
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(json.dumps([hashlib.sha256(t.encode()).hexdigest()
                                     for t in ("ABC김", "배제")]))
    monkeypatch.setattr(sys, "argv", ["prepare", "--source-dir", str(source),
                        "--existing-dir", str(existing), "--output", str(output),
                        "--exclude-text-hashes", str(exclusions), "--max-kdpii-train", "0"])
    preparation.main()
    final = json.loads((output / "manifest.json").read_text())
    assert final["test_data_opened"] is False
    assert final["cleaning"]["excluded_ids"] == ["kt2", "lt1"]
    assert final["outputs"]["train.jsonl"]["rows"] == 2
    assert final["outputs"]["kdpii-validation.jsonl"]["rows"] == 1
    assert final["outputs"]["klue-validation.jsonl"]["rows"] == 1
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    preparation.main()
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}
