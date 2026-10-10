"""Enforce frozen external evaluation gates without modifying the runtime.

Passing is limited to these reports. Historical regression contracts and other
release requirements remain unchecked; this script never promotes a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SCOPES = {"development": "development", "heldout": "heldout",
          "kdpii": "second_source_KDPII_frozen_candidate"}
REGRESSIONS = ("lost_correct_spans", "introduced_false_spans", "newly_exposed_names",
               "actual_stars_newly_exposed_names")
COUNTS = ("tp", "fp", "fn", "gold_names", "fully_covered_names",
          "unnecessary_masked_characters", "negative_false_positive_sentences")


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def evaluate_release(reports):
    """Return a fail-closed, machine-readable decision from supplied report dicts."""
    reasons, domains = [], []

    def reject(domain, check, detail):
        reasons.append(dict(domain=domain, check=check, detail=detail))

    reference = reports.get("development", {})
    reference_selection = reference.get("selection")
    reference_factory = reference.get("candidate_factory")
    reference_baseline = reference.get("baseline", {})
    identity = (reference_baseline.get("model_id"), reference_baseline.get("revision"))
    for name, scope in SCOPES.items():
        report = reports.get(name)
        if not isinstance(report, dict):
            reject(name, "complete_frozen_run", "Required completed report is absent")
            continue
        if report.get("scope") != scope or report.get("source_changed_during_run") is not False:
            reject(name, "complete_frozen_run", "Wrong scope or inputs changed during execution")
        if report.get("exact_training_or_validation_text_overlap_ids") != []:
            reject(name, "training_overlap", "Exact text overlap was not checked or is nonempty")
        selection = report.get("selection")
        if (not isinstance(selection, dict) or selection.get("finalized") is not True
                or not isinstance(selection.get("candidate"), dict) or not selection["candidate"]
                or selection != reference_selection or not reference_factory
                or report.get("candidate_factory") != reference_factory):
            reject(name, "same_selection", "Reports must use one identical finalized candidate")
        baseline = report.get("baseline", {})
        if not all(identity) or (baseline.get("model_id"), baseline.get("revision")) != identity:
            reject(name, "same_baseline", "Baseline model identity or revision differs/is absent")
        for label, hashes in (("report", report.get("frozen_sha256")),
                              ("selection", selection.get("frozen_sha256")
                               if isinstance(selection, dict) else None)):
            if not isinstance(hashes, dict) or not hashes:
                reject(name, "frozen_hashes", f"{label}: no frozen inputs provided")
                continue
            for filename, expected in hashes.items():
                try:
                    matches = digest(filename) == expected
                except (OSError, TypeError, ValueError):
                    matches = False
                if not matches:
                    reject(name, "frozen_hashes", f"{label}: input missing or changed: {filename}")
        policies = ("PS_NAME", "PS_NAME+PS_NICKNAME") if name == "kdpii" else (None,)
        for policy in policies:
            domain = f"{name}/{policy}" if policy else name
            domains.append(domain)
            try:
                candidate = report["candidate"]
                old, new = baseline["metrics"], candidate["metrics"]
                if policy:
                    old, new = old[policy], new[policy]
                    gate = candidate["comparisons"][policy]["regression_gate"]
                else:
                    gate = candidate["regression_gate"]
            except (KeyError, TypeError):
                reject(domain, "complete_metrics", "Required policy metrics or gate is absent")
                continue
            if (not isinstance(gate, dict) or gate.get("passed") is not True
                    or any(gate.get(field) != [] for field in REGRESSIONS)):
                reject(domain, "strict_regression_gate",
                       "Gate failed, required checks are absent, or concrete regressions remain")
            valid = True
            for label, score in (("baseline", old), ("candidate", new)):
                if (not isinstance(score, dict)
                        or any(type(score.get(key)) is not int or score[key] < 0 for key in COUNTS)
                        or type(score.get("f1")) not in (int, float)
                        or not math.isfinite(score["f1"]) or not 0 <= score["f1"] <= 1
                        or score["gold_names"] != score["tp"] + score["fn"]
                        or score["fully_covered_names"] > score["gold_names"]):
                    valid = False
                    reject(domain, "valid_metrics", f"{label}: missing/nonfinite/invalid metrics")
                    continue
                computed_f1 = (2 * score["tp"] / (2 * score["tp"] + score["fp"] + score["fn"])
                               if score["tp"] else 0.)
                if not math.isclose(score["f1"], computed_f1, rel_tol=1e-12, abs_tol=1e-12):
                    valid = False
                    reject(domain, "valid_metrics", f"{label}: F1 disagrees with exact counts")
            if not valid:
                continue
            if old["gold_names"] != new["gold_names"]:
                reject(domain, "same_gold", "Baseline and candidate have different gold counts")
            failed = [key for key in ("f1", "fully_covered_names") if new[key] < old[key]]
            failed.extend(key for key in ("negative_false_positive_sentences",
                                          "unnecessary_masked_characters") if new[key] > old[key])
            if failed:
                reject(domain, "aggregate_constraints", "Worse than baseline: " + ", ".join(failed))
    return dict(
        scope="external evaluation gate only", checked_domains=domains,
        external_gate_passed=not reasons, reasons=reasons,
        historical_contracts_checked=False, release_qualified=False,
        automatic_runtime_promotion=False,
        remaining_release_requirements=["Existing historical regression contracts not evaluated"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in SCOPES:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Exit nonzero if external gate fails")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output; existing release evidence cannot be overwritten")
    reports = {name: json.loads(getattr(args, name).read_text()) for name in SCOPES}
    decision = evaluate_release(reports)
    decision["checker_sha256"] = digest(__file__)
    decision["report_sha256"] = {str(getattr(args, name)): digest(getattr(args, name))
                                  for name in SCOPES}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    if args.check and not decision["external_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
