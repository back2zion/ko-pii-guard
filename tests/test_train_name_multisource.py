"""Training-only masking and exact frozen-feature reuse invariants."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from name_generalization_model import GeneralNameHead
    from train_name_generalization import MODEL_ID, REVISION, gold_labels
    from train_name_multisource import (
        apply_ignore_labels,
        case_key,
        index_frozen_cache,
        source_macro_rank,
        train_step,
        verify_encoding_archive,
    )
finally:
    sys.path[:] = previous_path


def case(identifier="a"):
    return dict(id=identifier, text="김 별", expected=[dict(entity="KR_NAME", start=0, end=1)],
                training_ignore_spans=[dict(label="PS_NICKNAME", start=2, end=3)])


def test_ignore_alias_labels_without_changing_original_cached_targets():
    labels = gold_labels(case())
    result = apply_ignore_labels(case(), labels)
    assert labels.tolist() == [4, 0, 0]
    assert result.tolist() == [4, 0, -100]
    malformed = dict(case(), training_ignore_spans=[dict(start=0, end=1, label="PS_ID")])
    with pytest.raises(ValueError, match="overlap"):
        apply_ignore_labels(malformed, labels)


def test_feature_keys_ignore_ids_but_require_exact_text_and_gold():
    assert case_key(case("one")) == case_key(case("two"))
    assert case_key(case()) != case_key(dict(case(), text="김  별"))
    assert case_key(case()) != case_key(dict(case(), expected=[]))


def test_frozen_cache_rejects_stale_labels_and_source_hashes(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(case()) + "\n")
    vocabulary = {"김": 2, " ": 3, "별": 4}
    chars = torch.tensor([2, 3, 4])
    row = (torch.zeros(3, 768), chars, torch.zeros(3, 2), torch.ones(3, 5),
           gold_labels(case()))
    key = dict(files={str(source): hashlib.sha256(source.read_bytes()).hexdigest()},
               encoder=[MODEL_ID, REVISION], vocabulary=vocabulary,
               encoding_source=None, character_encoding_source=None)
    payload = dict(key=key, rows=[row])
    indexed = index_frozen_cache(payload, vocabulary, verify_code=False)
    assert torch.equal(indexed[case_key(case())][-1], row[-1])
    payload["rows"] = [(*row[:-1], torch.zeros(3, dtype=torch.long))]
    with pytest.raises(ValueError, match="labels"):
        index_frozen_cache(payload, vocabulary, verify_code=False)
    payload["rows"] = [row]
    source.write_text(source.read_text() + " ")
    with pytest.raises(ValueError, match="hash"):
        index_frozen_cache(payload, vocabulary, verify_code=False)


def test_source_macro_selection_does_not_pool_large_domain_counts():
    first = {"wiki": dict(f1=.9, fp=5), "dialogue": dict(f1=.2, fp=50)}
    second = {"wiki": dict(f1=.8, fp=8), "dialogue": dict(f1=.8, fp=8)}
    assert source_macro_rank(second, .7, 2) > source_macro_rank(first, .9, 1)


def test_archived_encoding_hash_and_ast_equivalence_are_both_required(tmp_path):
    archive, current = tmp_path / "original.py.source.txt", tmp_path / "current.py"
    original = ("def encode(x):\n    return x\ndef gold_labels(x):\n    return x\n"
                "def main():\n    pass\n")
    archive.write_text(original)
    current.write_text(original.replace("    pass", "    print('new cache guards')"))
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert verify_encoding_archive(current, archive, expected)["method"] == "verified_archive_ast"
    archive.write_text(original + "# modified archive\n")
    with pytest.raises(ValueError, match="archive hash"):
        verify_encoding_archive(current, archive, expected)
    archive.write_text(original)
    current.write_text(original.replace("return x", "return x + 1", 1))
    with pytest.raises(ValueError, match="AST"):
        verify_encoding_archive(current, archive, expected)


def test_legacy_cache_character_hash_requires_original_manifest_proof(tmp_path, monkeypatch):
    import train_name_multisource as training

    original = tmp_path / "original"
    original.mkdir()
    trainer = Path(training.__file__).with_name("train_name_generalization.py")
    character = Path(training.__file__).with_name("name_generalization_model.py")
    trainer_sha = hashlib.sha256(trainer.read_bytes()).hexdigest()
    character_sha = hashlib.sha256(character.read_bytes()).hexdigest()
    archive = original / "train_name_generalization.py.source.txt"
    archive.write_bytes(trainer.read_bytes())
    (original / "name_generalization_model.py.source.txt").write_bytes(character.read_bytes())
    manifest = dict(source_sha256={str(trainer.resolve()): trainer_sha,
                                   str(character.resolve()): character_sha})
    (original / "manifest.json").write_text(json.dumps(manifest))
    payload = dict(key=dict(encoder=[MODEL_ID, REVISION], vocabulary={},
                           files={}, encoding_source=trainer_sha), rows=[])
    assert index_frozen_cache(payload, {}, source_archive=archive) == {}
    manifest["source_sha256"][str(character.resolve())] = "0" * 64
    (original / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="character"):
        index_frozen_cache(payload, {}, source_archive=archive)


def test_head_training_does_not_mutate_frozen_features_or_prior():
    torch.manual_seed(10)
    head = GeneralNameHead(char_vocab_size=5, hidden_size=4, projection_size=2,
                           char_embedding_size=2, lstm_hidden_size=2)
    row = (torch.randn(3, 4).half(), torch.tensor([2, 3, 4]), torch.zeros(3, 2),
           torch.full((3, 5), .2), torch.tensor([4, 0, -100]))
    before = [tensor.clone() for tensor in row]
    state = {key: value.clone() for key, value in head.state_dict().items()}
    optimizer = torch.optim.AdamW(head.parameters(), lr=.0005)
    loss = train_step(head, optimizer, [row])
    assert loss > 0
    assert all(torch.equal(a, b) for a, b in zip(before, row, strict=True))
    assert all(tensor.grad is None for tensor in row)
    assert any(not torch.equal(value, state[key]) for key, value in head.state_dict().items())
