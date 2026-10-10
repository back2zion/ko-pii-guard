"""Text-only amendment preserves ordering and cannot silently resample."""

import copy
import json
import sys
from pathlib import Path

import pytest

previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import name_reserve_text_audit as audit
finally:
    sys.path[:] = previous_path


def test_cross_domain_consumed_and_retained_duplicates_keep_original_first():
    reserve = {
        "klue": {"consumed_ids": ["old-k"], "selected_ids": ["a", "b", "c"]},
        "kdpii": {"consumed_ids": ["old-d"], "selected_ids": ["d", "e", "f", "g"]},
    }
    texts = {"klue": {"old-k": "old", "a": "Ａ씨", "b": "old-d", "c": "new"},
             "kdpii": {"old-d": "old-d", "d": "A씨", "e": "old", "f": "fresh",
                       "g": "fresh"}}
    result = audit.select_text_unique(reserve, texts)
    assert result["klue"]["selected_ids"] == ["a", "c"]
    assert result["kdpii"]["selected_ids"] == ["f"]
    assert result["kdpii"]["excluded_ids"] == ["d", "e", "g"]
    reasons = {row["id"]: row["reasons"] for row in result["kdpii"]["exclusions"]}
    assert reasons["d"] == ["reserve_normalized_duplicate"]
    assert "consumed_exact_duplicate" in reasons["e"]
    assert reasons["g"] == ["reserve_exact_duplicate", "reserve_normalized_duplicate"]
    assert "Ａ씨" not in json.dumps(result, ensure_ascii=False)


def test_runtime_normalization_is_used_without_broad_nfkc():
    reserve = {"klue": {"consumed_ids": [], "selected_ids": ["a", "b"]},
               "kdpii": {"consumed_ids": [], "selected_ids": []}}
    result = audit.select_text_unique(reserve, {"klue": {"a": "①", "b": "1"}, "kdpii": {}})
    assert result["klue"]["selected_ids"] == ["a", "b"]


def test_readers_do_not_interpret_gold_and_preserve_character_spaces(tmp_path):
    klue = tmp_path / "dev.tsv"
    klue.write_text("## klue-ner-a-nsmc\tHEADER IS NOT THE TEXT\n"
                    "홍\tNOT_A_BIO_TAG\n \tINVALID\n길\tUNREAD\n\n")
    kdpii = tmp_path / "test.json"
    kdpii.write_text(json.dumps([{"sent_idx": "a", "sentence": "original text",
                                  "PII_set": {"deliberately": "invalid annotation"}}]))
    assert audit.read_klue_texts(klue, ["klue-ner-a-nsmc"]) == {
        "klue-ner-a-nsmc": "홍 길"}
    assert audit.read_kdpii_texts(kdpii, ["a"]) == {"a": "original text"}


@pytest.mark.parametrize("reader,suffix,content", [
    ("read_klue_texts", ".tsv", "## klue-ner-a-nsmc\tx\n가\tx\n\n"
     "## klue-ner-a-nsmc\ty\n나\tx\n"),
    ("read_kdpii_texts", ".json", json.dumps([
        {"sent_idx": "klue-ner-a-nsmc", "sentence": "a"},
        {"sent_idx": "klue-ner-a-nsmc", "sentence": "b"}])),
])
def test_duplicate_target_ids_fail_closed(tmp_path, reader, suffix, content):
    path = tmp_path / ("source" + suffix)
    path.write_text(content)
    with pytest.raises(ValueError, match="duplicate|Duplicate"):
        getattr(audit, reader)(path, ["klue-ner-a-nsmc"])


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ORIGINAL_COUNTS", {"klue": 1, "kdpii": 2})
    klue = tmp_path / "klue-ner-v1.1_dev.tsv"
    klue.write_text("## klue-ner-a-nsmc\tx\n가\tINVALID\n\n")
    kdpii = tmp_path / "test.json"
    kdpii.write_text(json.dumps([{"sent_idx": "a", "sentence": "가"},
                                 {"sent_idx": "b", "sentence": "new"}]))
    reserve = tmp_path / "reserve.json"
    reserve.write_text(json.dumps({
        "scope": "v2_evaluation_reserve_before_any_new_training", "test_labels_read": False,
        "klue": {"selected_ids": ["klue-ner-a-nsmc"], "consumed_ids": []},
        "kdpii": {"selected_ids": ["a", "b"], "consumed_ids": []},
        "source_sha256": {str(klue): audit.digest(klue), str(kdpii): audit.digest(kdpii)},
    }))
    return reserve, klue, kdpii


def test_amendment_recomputed_and_pins_prevent_source_or_selection_tampering(tmp_path, monkeypatch):
    reserve, _, kdpii = fixture(tmp_path, monkeypatch)
    result = audit.build_amendment(reserve)
    output = tmp_path / "amendment.json"
    output.write_text(json.dumps(result))
    assert audit.validate_amendment(output, reserve)["domains"]["kdpii"]["selected_ids"] == ["b"]
    altered = copy.deepcopy(result)
    altered["domains"]["kdpii"]["selected_ids"] = ["a", "b"]
    output.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="recomputed"):
        audit.validate_amendment(output, reserve)
    output.write_text(json.dumps(result))
    kdpii.write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        audit.validate_amendment(output, reserve)


def test_missing_source_id_and_wrong_original_population_are_blocked(tmp_path, monkeypatch):
    reserve, klue, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="absent"):
        audit.read_klue_texts(klue, ["missing"])
    document = json.loads(reserve.read_text())
    document["kdpii"]["selected_ids"] = ["a"]
    reserve.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="population"):
        audit.build_amendment(reserve)
