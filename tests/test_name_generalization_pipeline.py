"""Do not open fresh data after stale selection or misleading aggregate replay."""

import copy
import hashlib
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from validate_name_generalization_pipeline import (
        validate_live_development,
        verify_policy_provenance,
    )
finally:
    sys.path[:] = previous_path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_files(tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"original model")
    cache = tmp_path / "development.pt"
    torch.save({"frozen_sha256": {str(weights): digest(weights)}}, cache)
    policy = {"source_changed_during_run": False,
              "source_sha256": {str(cache): digest(cache)}}
    return policy, cache, weights


def test_policy_validates_current_cached_model_inputs_not_only_old_success_flag(tmp_path):
    policy, _, weights = policy_files(tmp_path)
    verify_policy_provenance(policy)
    weights.write_bytes(b"different model after selection")
    with pytest.raises(ValueError, match="Cached development input.*changed"):
        verify_policy_provenance(policy)


def test_policy_rejects_changed_cache_even_if_new_cache_is_internally_consistent(tmp_path):
    policy, cache, weights = policy_files(tmp_path)
    weights.write_bytes(b"different model")
    torch.save({"frozen_sha256": {str(weights): digest(weights)}}, cache)
    with pytest.raises(ValueError, match="Policy input.*changed"):
        verify_policy_provenance(policy)


def score():
    return dict(tp=8, fp=1, fn=2, f1=16 / 19, fully_covered_names=9,
                unnecessary_masked_characters=1, negative_false_positive_sentences=1,
                errors=[dict(id="sentence-a", missed=[[0, 1]], false=[[2, 3]])])


def report(candidate, baseline):
    return dict(scope="development", source_changed_during_run=False,
                candidate={"metrics": candidate}, baseline={"metrics": baseline})


def test_same_aggregates_cannot_hide_candidate_errors_moving_to_other_sentences():
    expected, baseline = score(), score()
    actual = copy.deepcopy(expected)
    actual["errors"][0]["id"] = "sentence-b"
    with pytest.raises(ValueError, match="candidate.*replay"):
        validate_live_development(report(actual, baseline), expected, baseline)


def test_live_baseline_must_match_selection_baseline_per_case():
    expected, baseline = score(), score()
    actual_baseline = copy.deepcopy(baseline)
    actual_baseline["errors"][0]["missed"] = [[4, 5]]
    with pytest.raises(ValueError, match="baseline"):
        validate_live_development(report(expected, actual_baseline), expected, baseline)


def test_live_replay_must_still_satisfy_predeclared_selection_rank():
    expected, baseline = score(), score()
    validate_live_development(report(expected, baseline), expected, baseline)
    baseline["fully_covered_names"] += 1
    with pytest.raises(ValueError, match="criteria"):
        validate_live_development(report(expected, baseline), expected, baseline)
