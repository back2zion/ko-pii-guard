"""Domain failures and exact masking regressions cannot be pooled away."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/name_multisource_selection.py"
spec = importlib.util.spec_from_file_location("multisource_selection", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
select_candidates = module.select_candidates


def score(tp=8, fp=1, fn=2, full=8, extra=2, negative_fp=1):
    return dict(tp=tp, fp=fp, fn=fn, gold_names=tp + fn,
                f1=2 * tp / (2 * tp + fp + fn) if tp else 0.,
                fully_covered_names=full, unnecessary_masked_characters=extra,
                negative_false_positive_sentences=negative_fp)


def gate():
    return dict(passed=True, lost_correct_spans=[], introduced_false_spans=[],
                newly_exposed_names=[], actual_stars_newly_exposed_names=[])


def candidate(identifier, scores):
    return dict(id=identifier, policy={"cutoff": 0.5}, sources={
        source: dict(metrics=copy.deepcopy(metrics), regression_gate=gate())
        for source, metrics in scores.items()})


def test_everywhere_equivalent_is_no_candidate_not_fake_improvement():
    baselines = {"news": score(), "dialogue": score()}
    result = select_candidates(baselines, [candidate("same", baselines)],
                               required_sources=list(baselines))
    assert result["status"] == "no_candidate"
    assert result["selected_id"] is None
    assert result["automatic_runtime_promotion"] is False


def test_strict_improvement_on_one_source_with_every_gate_passing_can_be_selected():
    baselines = {"news": score(), "dialogue": score()}
    improved = {"news": score(tp=9, fn=1, full=9), "dialogue": score()}
    rows = [candidate("z", improved), candidate("a", improved)]
    first = select_candidates(baselines, rows, required_sources=list(baselines))
    second = select_candidates(baselines, rows[::-1], required_sources=list(baselines))
    assert first["status"] == "selected"
    assert first["selected_id"] == second["selected_id"] == "a"
    assert first["feasible_ids"] == second["feasible_ids"]


def test_measured_kdpii_false_alarms_cannot_hide_behind_total_coverage_improvement():
    baselines = {
        "klue": score(765, 126, 143, 812, 217, 43),
        "kdpii/PS_NAME": score(15, 2, 3, 16, 5, 1),
    }
    measured = {
        "klue": score(810, 89, 98, 823, 118, 19),
        "kdpii/PS_NAME": score(17, 116, 1, 18, 168, 101),
    }
    assert sum(s["fully_covered_names"] for s in measured.values()) > sum(
        s["fully_covered_names"] for s in baselines.values())
    result = select_candidates(baselines, [candidate("measured", measured)],
                               required_sources=list(baselines))
    assert result["status"] == "no_candidate"
    assert any(r["source"] == "kdpii/PS_NAME" and r["check"] == "aggregate_constraints"
               for r in result["rejections"]["measured"])


@pytest.mark.parametrize("missing", ["baseline", "source", "gate"])
def test_missing_baseline_source_or_gate_fails_closed(missing):
    baselines = {"news": score(), "dialogue": score()}
    row = candidate("incomplete", {"news": score(tp=9, fn=1, full=9), "dialogue": score()})
    if missing == "baseline":
        del baselines["dialogue"]
    elif missing == "source":
        del row["sources"]["dialogue"]
    else:
        del row["sources"]["dialogue"]["regression_gate"]
    result = select_candidates(baselines, [row], required_sources=["news", "dialogue"])
    assert result["status"] == "no_candidate"


def test_actual_stars_regression_rejects_claimed_true_gate():
    baseline = {"news": score()}
    row = candidate("regressed", {"news": score(tp=9, fn=1, full=9)})
    row["sources"]["news"]["regression_gate"]["actual_stars_newly_exposed_names"] = [["id", 0, 2]]
    result = select_candidates(baseline, [row], required_sources=["news"])
    assert result["status"] == "no_candidate"


@pytest.mark.parametrize("field,value", [("f1", float("nan")), ("fp", -1),
                                         ("tp", 8.5), ("fully_covered_names", 100),
                                         ("f1", 1.0)])
def test_invalid_nonfinite_or_inconsistent_metrics_rejected(field, value):
    baseline = {"news": score()}
    row = candidate("invalid", {"news": score(tp=9, fn=1, full=9)})
    row["sources"]["news"]["metrics"][field] = value
    result = select_candidates(baseline, [row], required_sources=["news"])
    assert result["status"] == "no_candidate"


def test_duplicate_candidate_identity_is_rejected():
    baseline = {"news": score()}
    row = candidate("duplicate", {"news": score(tp=9, fn=1, full=9)})
    result = select_candidates(baseline, [row, copy.deepcopy(row)], required_sources=["news"])
    assert result["status"] == "no_candidate"


@pytest.mark.parametrize("field,value", [("sentences", 11), ("negative_sentences", 5),
                                         ("negative_sentences", None)])
def test_candidate_must_preserve_baseline_sentence_population(field, value):
    baseline = score()
    baseline.update(sentences=10, negative_sentences=4)
    improved = score(tp=9, fn=1, full=9)
    improved.update(sentences=10, negative_sentences=4)
    if value is None:
        del improved[field]
    else:
        improved[field] = value
    result = select_candidates({"news": baseline}, [candidate("different", {"news": improved})],
                               required_sources=["news"])
    assert result["status"] == "no_candidate"
    assert any(r["check"] == "same_population" for r in result["rejections"]["different"])


def test_check_cli_returns_nonzero_for_no_candidate(tmp_path):
    import hashlib

    frozen = tmp_path / "frozen.bin"
    frozen.write_bytes(b"source")
    baselines = {"news": score()}
    report = dict(scope="multisource_development_only", source_changed_during_run=False,
                  required_sources=["news"], baselines=baselines,
                  candidates=[candidate("unchanged", baselines)],
                  frozen_sha256={str(frozen): hashlib.sha256(frozen.read_bytes()).hexdigest()})
    path, output = tmp_path / "development.json", tmp_path / "decision.json"
    path.write_text(json.dumps(report))
    result = subprocess.run([sys.executable, str(SCRIPT), "--development", str(path),
                             "--output", str(output), "--check"], capture_output=True, text=True)
    assert result.returncode == 1
    assert json.loads(output.read_text())["status"] == "no_candidate"
