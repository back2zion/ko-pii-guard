"""Select only strictly improving development candidates passing every source.

No pooled score can waive a source's quality or per-case masking regression.
This module does not open fresh evaluation data or promote a runtime model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

REGRESSIONS = ("lost_correct_spans", "introduced_false_spans", "newly_exposed_names",
               "actual_stars_newly_exposed_names")
COUNTS = ("tp", "fp", "fn", "gold_names", "fully_covered_names",
          "unnecessary_masked_characters", "negative_false_positive_sentences")
LOWER = ("negative_false_positive_sentences", "unnecessary_masked_characters")
HIGHER = ("fully_covered_names", "f1")
RANKING = ("Maximize minimum source coverage-rate gain, mean coverage-rate gain, mean F1 "
           "gain, total negative-FP reduction, total extra-character reduction; lexical ID tie")


def metric_problem(score):
    if not isinstance(score, dict):
        return "Metrics are absent"
    if any(type(score.get(key)) is not int or score[key] < 0 for key in COUNTS):
        return "Required counts must be nonnegative integers"
    if (type(score.get("f1")) not in (int, float) or not math.isfinite(score["f1"])
            or not 0 <= score["f1"] <= 1):
        return "F1 must be a finite probability"
    if (score["gold_names"] != score["tp"] + score["fn"]
            or score["fully_covered_names"] > score["gold_names"]
            or score["negative_false_positive_sentences"] > score["fp"]):
        return "Counts are inconsistent"
    expected = (2 * score["tp"] / (2 * score["tp"] + score["fp"] + score["fn"])
                if score["tp"] else 0.)
    if not math.isclose(score["f1"], expected, rel_tol=1e-12, abs_tol=1e-12):
        return "F1 disagrees with exact counts"
    for key in ("sentences", "negative_sentences"):
        if key in score and (type(score[key]) is not int or score[key] < 0):
            return "Optional sentence counts must be nonnegative integers"
    if ("negative_sentences" in score
            and score["negative_false_positive_sentences"] > score["negative_sentences"]):
        return "Negative FP sentences exceed negative sentences"
    if ("sentences" in score and "negative_sentences" in score
            and score["negative_sentences"] > score["sentences"]):
        return "Negative sentences exceed all sentences"
    return None


def _decision(required_sources, errors, rejections, feasible):
    ranked = sorted(feasible, key=lambda row: (*(-value for value in row["rank"]), row["id"]))
    selected = ranked[0] if ranked and not errors else None
    return dict(
        scope="multisource development selection only",
        status="selected" if selected else "no_candidate",
        selected_id=selected["id"] if selected else None,
        selected_policy=selected["policy"] if selected else None,
        required_sources=required_sources, ranking_rule=RANKING,
        feasible_ids=[row["id"] for row in ranked] if not errors else [],
        feasible_ranks={row["id"]: list(row["rank"]) for row in ranked} if not errors else {},
        global_reasons=errors, rejections=rejections,
        automatic_runtime_promotion=False,
    )


def select_candidates(baselines, candidates, *, required_sources):
    """Fail closed on any missing source, invalid score, regression, or no improvement."""
    if (not isinstance(required_sources, (list, tuple)) or not required_sources
            or any(not isinstance(source, str) or not source for source in required_sources)
            or len(set(required_sources)) != len(required_sources)):
        return _decision([], ["Required source names must be nonempty and unique"], {}, [])
    sources = sorted(required_sources)
    errors, rejected, feasible = [], {}, []
    if not isinstance(baselines, dict) or set(baselines) != set(sources):
        errors.append("Baselines must contain exactly every required source")
    else:
        for source in sources:
            problem = metric_problem(baselines[source])
            if problem:
                errors.append(f"Baseline {source}: {problem}")
    if not isinstance(candidates, list):
        return _decision(sources, [*errors, "Candidates must be a list"], {}, [])
    identifiers = [row.get("id") for row in candidates if isinstance(row, dict)
                   and isinstance(row.get("id"), str)]
    duplicate_ids = sorted(key for key, count in Counter(identifiers).items() if count > 1)
    if duplicate_ids:
        errors.append("Duplicate candidate IDs: " + ", ".join(duplicate_ids))
    if errors:
        return _decision(sources, errors, {}, [])
    for index, row in enumerate(candidates):
        identifier = row.get("id") if isinstance(row, dict) else None
        if not isinstance(identifier, str) or not identifier:
            rejected[f"invalid_candidate_{index}"] = [dict(
                source=None, check="candidate_complete", detail="Nonempty candidate ID required")]
            continue
        reasons = []

        def reject(source, check, detail, target=reasons):
            target.append(dict(source=source, check=check, detail=detail))

        evaluations = row.get("sources")
        if not isinstance(row.get("policy"), dict):
            reject(None, "candidate_complete", "Explicit policy dictionary required")
        if not isinstance(evaluations, dict) or set(evaluations) != set(sources):
            reject(None, "source_complete", "Candidate must contain exactly every required source")
            rejected[identifier] = reasons
            continue
        improved, coverage_gains, f1_gains = False, [], []
        negative_gain = extra_gain = 0
        for source in sources:
            result = evaluations[source]
            if not isinstance(result, dict):
                reject(source, "source_complete", "Source result must contain metrics and gate")
                continue
            gate = result.get("regression_gate")
            if (not isinstance(gate, dict) or gate.get("passed") is not True
                    or any(gate.get(key) != [] for key in REGRESSIONS)):
                reject(source, "strict_regression_gate", "Missing/failed gate or regressions")
            score, baseline = result.get("metrics"), baselines[source]
            problem = metric_problem(score)
            if problem:
                reject(source, "valid_metrics", problem)
                continue
            if score["gold_names"] != baseline["gold_names"]:
                reject(source, "same_gold", "Candidate and baseline gold counts differ")
                continue
            changed_population = [key for key in ("sentences", "negative_sentences")
                                  if key in baseline and score.get(key) != baseline[key]]
            if changed_population:
                reject(source, "same_population", "Sentence counts missing or different: "
                       + ", ".join(changed_population))
                continue
            worse = [key for key in HIGHER if score[key] < baseline[key]]
            worse.extend(key for key in LOWER if score[key] > baseline[key])
            if worse:
                reject(source, "aggregate_constraints", "Worse than baseline: " + ", ".join(worse))
            improved |= any(score[key] > baseline[key] for key in HIGHER)
            improved |= any(score[key] < baseline[key] for key in LOWER)
            coverage_gains.append((score["fully_covered_names"] - baseline["fully_covered_names"])
                                  / baseline["gold_names"] if baseline["gold_names"] else 0.)
            f1_gains.append(score["f1"] - baseline["f1"])
            negative_gain += (baseline["negative_false_positive_sentences"]
                              - score["negative_false_positive_sentences"])
            extra_gain += baseline["unnecessary_masked_characters"] - score[
                "unnecessary_masked_characters"]
        if not improved:
            reject(None, "strict_improvement", "No required source strictly improves over baseline")
        if reasons:
            rejected[identifier] = reasons
            continue
        rank = (min(coverage_gains), sum(coverage_gains) / len(sources),
                sum(f1_gains) / len(sources), negative_gain, extra_gain)
        feasible.append(dict(id=identifier, policy=row["policy"], rank=rank))
    return _decision(sources, [], dict(sorted(rejected.items())), feasible)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path; prior decisions must not be overwritten")
    report = json.loads(args.development.read_text())
    provenance_errors = []
    if (report.get("scope") != "multisource_development_only"
            or report.get("source_changed_during_run") is not False):
        provenance_errors.append("Require completed multisource development with unchanged inputs")
    hashes = report.get("frozen_sha256")
    if not isinstance(hashes, dict) or not hashes:
        provenance_errors.append("Development report requires frozen input hashes")
    else:
        for path, expected in hashes.items():
            if not Path(path).is_file() or digest(path) != expected:
                provenance_errors.append(f"Frozen development input missing or changed: {path}")
    decision = (_decision(report.get("required_sources", []), provenance_errors, {}, [])
                if provenance_errors else select_candidates(
                    report.get("baselines"), report.get("candidates"),
                    required_sources=report.get("required_sources")))
    decision.update(development_report_sha256=digest(args.development),
                    checker_sha256=digest(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    if args.check and decision["status"] != "selected":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
