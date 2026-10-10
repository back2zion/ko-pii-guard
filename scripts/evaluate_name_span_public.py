"""Run frozen research heads through actual analyze and stars mask APIs.

Previously inspected synthetic evaluation is diagnostic development only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from name_span_adapter import JointSpanNER
from name_span_experiment import SpanNameHead
from safetensors.torch import load_file
from train_name_span_experiment import gate, metrics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from name_address_benchmark import load_cases  # noqa: E402

from ko_pii_guard import KoreanPIIGuard  # noqa: E402
from ko_pii_guard.ner import KoreanNER  # noqa: E402


def run(cases, guard):
    predictions, masks = [], []
    for case in cases:
        predictions.append(
            {(r.start, r.end) for r in guard.analyze(case["text"]) if r.entity == "KR_NAME"}
        )
        masked = guard.mask(case["text"], style="stars")
        if len(masked) != len(case["text"]):
            raise ValueError("Stars masking changed original coordinate system")
        masks.append(masked)
    score = metrics(cases, predictions)
    score["fully_covered_names"] = sum(
        masked[e["start"] : e["end"]] == "*" * (e["end"] - e["start"])
        for case, masked in zip(cases, masks, strict=True)
        for e in case["expected"]
        if e["entity"] == "KR_NAME"
    )
    return score, predictions, masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/name-context-v11")
    parser.add_argument(
        "--data", type=Path, default=ROOT / "benchmarks/data/name_context_v11.jsonl"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output; do not overwrite evaluations")
    selection_report = json.loads((args.experiment / "report.json").read_text())
    if selection_report.get("source_changed_during_run") is not False:
        raise ValueError("Training must finish before public API evaluation")
    torch.set_num_threads(2)
    cases = [c for c in load_cases(args.data) if c["split"] == "evaluation"]
    files = [
        Path(__file__),
        args.data,
        ROOT / "scripts/name_span_adapter.py",
        ROOT / "scripts/name_span_experiment.py",
        ROOT / "scripts/train_name_span_experiment.py",
        args.experiment / "report.json",
        *sorted((ROOT / "src/ko_pii_guard").glob("*.py")),
        *args.baseline.glob("*.safetensors"),
        args.baseline / "name_context_config.json",
        *[
            p
            for d in args.experiment.glob("seed-*")
            for p in (d / "head.safetensors", d / "config.json")
        ],
    ]

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    hashes = {str(p): digest(p) for p in files}
    ner = KoreanNER.from_pretrained(name_context_path=args.baseline)
    baseline_guard = KoreanPIIGuard(entities=["KR_NAME"], score_threshold=0.0, ner=ner)
    baseline_score, baseline, baseline_masks = run(cases, baseline_guard)
    report = dict(
        scope="previously_inspected_synthetic_public_API_diagnosis",
        guard_score_threshold=0.0,
        threshold_reason="Research head threshold controls acceptance; avoid a second gate",
        source_sha256=hashes,
        baseline=baseline_score,
        models={},
        runtime_promotion=False,
    )
    for directory in sorted(args.experiment.glob("seed-*")):
        config = json.loads((directory / "config.json").read_text())
        head = SpanNameHead(**config["architecture"])
        head.load_state_dict(load_file(str(directory / "head.safetensors")))
        adapter = JointSpanNER(
            ner,
            head,
            config["vocabulary"],
            max_span_width=config["max_span_width"],
            threshold=config["threshold"],
        )
        guard = KoreanPIIGuard(entities=["KR_NAME"], score_threshold=0.0, ner=adapter)
        score, predictions, masks = run(cases, guard)
        regression = gate(cases, baseline, predictions)
        actual_exposed = [
            (c["id"], e["start"], e["end"])
            for c, old, new in zip(cases, baseline_masks, masks, strict=True)
            for e in c["expected"]
            if e["entity"] == "KR_NAME"
            and old[e["start"] : e["end"]] == "*" * (e["end"] - e["start"])
            and new[e["start"] : e["end"]] != "*" * (e["end"] - e["start"])
        ]
        regression["actual_stars_newly_exposed_names"] = actual_exposed
        regression["passed"] &= not actual_exposed
        report["models"][directory.name] = dict(metrics=score, regression_gate=regression)
        print(
            directory.name,
            {k: v for k, v in score.items() if k != "errors"},
            "gate",
            regression["passed"],
            flush=True,
        )
    report["source_changed_during_run"] = any(digest(Path(p)) != h for p, h in hashes.items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["source_changed_during_run"]:
        raise SystemExit("Frozen inputs changed during public API evaluation")


if __name__ == "__main__":
    main()
