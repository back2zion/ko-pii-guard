"""Real-source split integrity and original character boundary guarantees."""

import sys
from pathlib import Path

import pytest

previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from prepare_name_generalization_data import parse_train, prepare, text_digest, verified_bytes
finally:
    sys.path[:] = previous_path


def row(identifier, text, expected=(), source="wikitree", split="train"):
    return dict(id=identifier, text=text, expected=list(expected), source=source, split=split)


def test_train_parser_preserves_spaces_and_adjacent_person_boundaries():
    content = (
        "## klue-ner-v1_train_00000_wikitree\tignored annotated header\n"
        "김\tB-PS\n \tO\n이\tB-PS\n나\tI-PS\n다\tB-PS\n"
    )
    (case,) = parse_train(content)
    assert case["text"] == "김 이나다"
    assert case["char_bio"] == ["B-PS", "O", "B-PS", "I-PS", "B-PS"]
    assert [(e["start"], e["end"]) for e in case["expected"]] == [(0, 1), (2, 4), (4, 5)]


@pytest.mark.parametrize(
    "identifier,tag",
    [
        ("klue-ner-v1_dev_00000-wikitree", "B-PS"),
        ("klue-ner-v1_train_00000_wikitree", "I-OG"),
    ],
)
def test_parser_rejects_dev_and_invalid_bio(identifier, tag):
    with pytest.raises(ValueError):
        parse_train(f"## {identifier}\theader\n김\t{tag}\n")


def test_duplicate_texts_cannot_cross_splits_and_selection_ignores_labels():
    cases = [
        row(f"{source}-{i}", f"{source}:{i}", source=source)
        for source in ("wikitree", "nsmc")
        for i in range(10)
    ]
    cases.append(row("duplicate", "wikitree:3"))
    train, validation, report = prepare(cases, [], validation_fraction=0.2)
    assert len(validation) == 4
    assert {c["source"] for c in validation} == {"wikitree", "nsmc"}
    assert not {c["text"] for c in train} & {c["text"] for c in validation}
    assert len(train) + len(validation) == 20
    changed = [dict(c, expected=[dict(entity="KR_NAME", start=0, end=1)]) for c in reversed(cases)]
    train2, validation2, _ = prepare(changed, [], validation_fraction=0.2)
    assert [c["id"] for c in train] == [c["id"] for c in train2]
    assert [c["id"] for c in validation] == [c["id"] for c in validation2]
    assert report["duplicate_rows_removed"] == 1


def test_synthetic_evaluation_and_explicit_exclusion_hashes_are_never_used():
    real = [row(str(i), str(i)) for i in range(10)]
    synthetic = [row("synthetic-train", "9"), row("old-eval", "0", split="evaluation")]
    train, validation, report = prepare(
        real, synthetic, excluded_hashes={text_digest("1")}, validation_fraction=0.2
    )
    assert all(c["text"] not in {"0", "1"} for c in train + validation)
    assert len({c["text"] for c in train + validation}) == len(train + validation)
    assert report["excluded_text_rows"] == 2


def test_conflicting_labels_are_quarantined_and_reported():
    real = [row(str(i), str(i)) for i in range(10)]
    real.append(row("bad-label", "2", [dict(entity="KR_NAME", start=0, end=1)]))
    train, validation, report = prepare(real, [], validation_fraction=0.2)
    assert all(c["text"] != "2" for c in train + validation)
    assert report["conflicting_text_groups"] == [["2", "bad-label"]]


def test_existing_download_hash_mismatch_is_not_overwritten(tmp_path):
    path = tmp_path / "source.tsv"
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="mismatch"):
        verified_bytes(path, "https://example.invalid", "0" * 64)
    assert path.read_bytes() == b"tampered"


def test_rejects_duplicate_ids_and_invalid_name_coordinates():
    with pytest.raises(ValueError, match="ID"):
        prepare([row("same", "a"), row("same", "b")], [])
    with pytest.raises(ValueError, match="span"):
        prepare([row("bad", "a", [dict(entity="KR_NAME", start=0, end=2)])], [])
