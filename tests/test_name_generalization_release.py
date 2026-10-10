"""External release gates must fail despite aggregate gains hiding regressions."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_name_generalization_release.py"
spec = importlib.util.spec_from_file_location("name_release", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
evaluate_release = module.evaluate_release


def fixture_reports(tmp_path):
    frozen = tmp_path / "weights.bin"
    frozen.write_bytes(b"fixed model")
    hashes = {str(frozen): hashlib.sha256(frozen.read_bytes()).hexdigest()}
    score = dict(tp=9, fp=1, fn=1, f1=0.9, fully_covered_names=9,
                 unnecessary_masked_characters=2, negative_false_positive_sentences=1,
                 gold_names=10)
    gate = dict(passed=True, lost_correct_spans=[], introduced_false_spans=[],
                newly_exposed_names=[], actual_stars_newly_exposed_names=[])
    selection = dict(finalized=True, candidate={"checkpoint": "/fixed", "threshold": 0.1},
                     frozen_sha256=hashes)
    reports = {}
    for name, scope in (("development", "development"), ("heldout", "heldout"),
                        ("kdpii", "second_source_KDPII_frozen_candidate")):
        report = dict(scope=scope, source_changed_during_run=False,
                      exact_training_or_validation_text_overlap_ids=[], frozen_sha256=hashes,
                      selection=selection, candidate_factory="candidate:build_ner",
                      baseline={"model_id": "baseline", "revision": "fixed",
                                "metrics": copy.deepcopy(score)},
                      candidate={"metrics": copy.deepcopy(score), "regression_gate": gate})
        if name == "kdpii":
            report["baseline"]["metrics"] = {
                p: copy.deepcopy(score) for p in ("PS_NAME", "PS_NAME+PS_NICKNAME")}
            report["candidate"] = dict(
                metrics={p: copy.deepcopy(score) for p in ("PS_NAME", "PS_NAME+PS_NICKNAME")},
                comparisons={p: {"regression_gate": copy.deepcopy(gate)}
                             for p in ("PS_NAME", "PS_NAME+PS_NICKNAME")})
        reports[name] = copy.deepcopy(report)
    return reports, frozen


def test_all_external_passes_still_do_not_qualify_historical_release(tmp_path):
    reports, _ = fixture_reports(tmp_path)
    decision = evaluate_release(reports)
    assert decision["external_gate_passed"]
    assert decision["scope"] == "external evaluation gate only"
    assert decision["release_qualified"] is False
    assert decision["automatic_runtime_promotion"] is False
    assert decision["historical_contracts_checked"] is False


def test_aggregate_gain_does_not_waive_newly_exposed_name(tmp_path):
    reports, _ = fixture_reports(tmp_path)
    candidate = reports["heldout"]["candidate"]
    candidate["metrics"]["fully_covered_names"] = 10
    candidate["regression_gate"]["actual_stars_newly_exposed_names"] = [["old", 0, 2]]
    # Even an incorrectly retained passed=True must not override concrete regressions.
    decision = evaluate_release(reports)
    assert not decision["external_gate_passed"]
    assert any(r["check"] == "strict_regression_gate" for r in decision["reasons"])


def test_each_kdpii_policy_must_pass_aggregate_constraints(tmp_path):
    reports, _ = fixture_reports(tmp_path)
    broad = reports["kdpii"]["candidate"]["metrics"]["PS_NAME+PS_NICKNAME"]
    broad["unnecessary_masked_characters"] = 3
    decision = evaluate_release(reports)
    assert not decision["external_gate_passed"]
    assert any(r["domain"] == "kdpii/PS_NAME+PS_NICKNAME"
               and r["check"] == "aggregate_constraints" for r in decision["reasons"])


def test_changed_frozen_file_blocks_completed_report(tmp_path):
    reports, frozen = fixture_reports(tmp_path)
    frozen.write_bytes(b"changed model")
    decision = evaluate_release(reports)
    assert not decision["external_gate_passed"]
    assert any(r["check"] == "frozen_hashes" for r in decision["reasons"])


def test_different_candidate_or_overlap_cannot_be_combined_as_one_release(tmp_path):
    reports, _ = fixture_reports(tmp_path)
    reports["kdpii"]["selection"]["candidate"]["threshold"] = 0.2
    reports["heldout"]["exact_training_or_validation_text_overlap_ids"] = ["duplicate"]
    decision = evaluate_release(reports)
    assert not decision["external_gate_passed"]
    assert {r["check"] for r in decision["reasons"]} >= {"same_selection", "training_overlap"}


def test_nan_metrics_and_incomplete_reports_are_rejected(tmp_path):
    reports, _ = fixture_reports(tmp_path)
    reports["heldout"]["source_changed_during_run"] = True
    reports["development"]["candidate"]["metrics"]["f1"] = float("nan")
    decision = evaluate_release(reports)
    assert not decision["external_gate_passed"]
    assert {r["check"] for r in decision["reasons"]} >= {"complete_frozen_run", "valid_metrics"}


def test_report_cannot_claim_f1_inconsistent_with_exact_counts(tmp_path):
    reports, _ = fixture_reports(tmp_path)
    reports["heldout"]["candidate"]["metrics"]["f1"] = 0.99
    decision = evaluate_release(reports)
    assert not decision["external_gate_passed"]
    assert any(r["check"] == "valid_metrics" for r in decision["reasons"])


def test_check_cli_writes_decision_and_exits_nonzero_on_regression(tmp_path):
    reports, _ = fixture_reports(tmp_path)
    reports["heldout"]["candidate"]["regression_gate"]["passed"] = False
    arguments = []
    for name, report in reports.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(report))
        arguments.extend([f"--{name}", str(path)])
    output = tmp_path / "decision.json"
    result = subprocess.run([sys.executable, str(SCRIPT), *arguments,
                             "--output", str(output), "--check"], capture_output=True, text=True)
    assert result.returncode != 0
    assert json.loads(output.read_text())["external_gate_passed"] is False
