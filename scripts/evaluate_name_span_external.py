"""Frozen-checkpoint external PS-boundary stress test on official KLUE dev data.

No downloads or training here. Raw external text remains outside this repository.
This is person NER, not a measure of private-person masking policy correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

import torch
from name_span_adapter import JointSpanNER
from name_span_experiment import SpanNameHead
from safetensors.torch import load_file
from train_name_span_experiment import gate, metrics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from ko_pii_guard import KoreanPIIGuard  # noqa: E402
from ko_pii_guard.ner import MODEL_ID, MODEL_REVISION, KoreanNER  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_klue(content):
    cases, chars, tags, identifier = [], [], [], None

    def flush():
        if identifier is None:
            return
        expected, active = [], None
        for i, tag in enumerate([*tags, "O"]):
            if tag != "I-PS" and active is not None:
                expected.append(dict(entity="KR_NAME", start=active, end=i))
                active = None
            if tag == "B-PS":
                active = i
            elif tag == "I-PS" and active is None:
                raise ValueError(f"Orphan I-PS: {identifier}")
        cases.append(
            dict(
                id=identifier,
                text="".join(chars),
                expected=expected,
                track="external_klue_person",
                split="evaluation",
            )
        )

    for line in content.splitlines():
        if line.startswith("## klue-ner-"):
            flush()
            identifier, chars, tags = line[3:].split("\t", 1)[0], [], []
        elif line.startswith("##") or not line:
            continue
        else:
            if identifier is None:
                raise ValueError("Character before document header")
            char, tag = line.rsplit("\t", 1)
            if len(char) != 1 or tag not in {
                "O",
                *[f"{p}-{t}" for p in ("B", "I") for t in ("PS", "OG", "LC", "DT", "TI", "QT")],
            }:
                raise ValueError(f"Invalid character annotation: {identifier}")
            chars.append(char)
            tags.append(tag)
    flush()
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Empty corpus or duplicate document ID")
    return cases


def select_cases(cases, count):
    if not 1 <= count <= len(cases):
        raise ValueError("Invalid sample size")
    # Fixed hash ranking, no balance/length/error-dependent filtering.
    return sorted(
        cases, key=lambda c: hashlib.sha256(("ko-pii-external-v1:" + c["id"]).encode()).hexdigest()
    )[:count]


def detailed_metrics(cases, predictions):
    score = metrics(cases, predictions)
    score["negative_sentences"] = sum(not c["expected"] for c in cases)
    score["negative_false_positive_sentences"] = sum(
        not case["expected"] and bool(predicted)
        for case, predicted in zip(cases, predictions, strict=True)
    )
    score["by_source"] = {}
    for source in sorted({c["id"].rsplit("-", 1)[-1] for c in cases}):
        indices = [i for i, c in enumerate(cases) if c["id"].rsplit("-", 1)[-1] == source]
        result = metrics([cases[i] for i in indices], [predictions[i] for i in indices])
        score["by_source"][source] = {k: v for k, v in result.items() if k != "errors"}
    return score


def paired_f1_interval(cases, baseline, predictions, *, repetitions=500):
    """Paired sentence bootstrap; original article/thread groups are unavailable."""
    counts = []
    for case, old, new in zip(cases, baseline, predictions, strict=True):
        gold = {(e["start"], e["end"]) for e in case["expected"] if e["entity"] == "KR_NAME"}
        counts.append(
            tuple(
                v
                for predicted in (old, new)
                for v in (len(gold & predicted), len(predicted - gold), len(gold - predicted))
            )
        )
    rng, deltas = random.Random(20261010), []
    for _ in range(repetitions):
        total = [0] * 6
        for _ in cases:
            for i, value in enumerate(counts[rng.randrange(len(counts))]):
                total[i] += value
        scores = [
            2 * total[i] / (2 * total[i] + total[i + 1] + total[i + 2]) if total[i] else 0.0
            for i in (0, 3)
        ]
        deltas.append(scores[1] - scores[0])
    deltas.sort()
    return dict(
        method="paired sentence percentile bootstrap",
        repetitions=repetitions,
        seed=20261010,
        f1_delta_95_interval=[
            deltas[int(0.025 * (repetitions - 1))],
            deltas[int(0.975 * (repetitions - 1))],
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/ko-pii-klue-ner"))
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/name-context-v11")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Do not overwrite an external evaluation")
    torch.set_num_threads(2)
    source = json.loads((args.source_dir / "source.json").read_text())
    for filename, metadata in source["files"].items():
        if digest(args.source_dir / filename) != metadata["sha256"]:
            raise ValueError("External source changed")
    all_cases = parse_klue((args.source_dir / "klue-ner-v1.1_dev.tsv").read_text())
    cases = select_cases(all_cases, args.sample_size)
    development = [
        json.loads(line)
        for line in (ROOT / "benchmarks/data/name_context_v11.jsonl").read_text().splitlines()
    ]
    development_texts = {c["text"] for c in development}
    overlap_ids = [c["id"] for c in cases if c["text"] in development_texts]
    selection_report = json.loads((args.experiment / "report.json").read_text())
    if selection_report.get("source_changed_during_run") is not False:
        raise ValueError("Training must be complete with unchanged frozen inputs")
    expected_seeds = selection_report["manifest"]["seeds"]
    if sorted(d.name for d in args.experiment.glob("seed-*")) != sorted(
        f"seed-{seed}" for seed in expected_seeds
    ):
        raise ValueError("Not all predeclared seeds have frozen checkpoints")
    model_files = [
        p
        for directory in sorted(args.experiment.glob("seed-*"))
        for p in (directory / "head.safetensors", directory / "config.json")
    ]
    if not model_files or not (args.experiment / "report.json").exists():
        raise ValueError("All checkpoints must be selected before external evaluation")
    files = [
        Path(__file__),
        args.source_dir / "source.json",
        *[args.source_dir / name for name in source["files"]],
        ROOT / "scripts/name_span_experiment.py",
        ROOT / "scripts/name_span_adapter.py",
        ROOT / "scripts/train_name_span_experiment.py",
        ROOT / "scripts/train_name_context.py",
        args.experiment / "report.json",
        ROOT / "benchmarks/data/name_context_v11.jsonl",
        *model_files,
        *args.baseline.glob("*.safetensors"),
        args.baseline / "name_context_config.json",
        *sorted((ROOT / "src/ko_pii_guard").glob("*.py")),
    ]
    hashes = {str(p): digest(p) for p in files}
    report = dict(
        scope="external_person_boundary_stress_test",
        source=source,
        license="CC-BY-SA-4.0; KLUE authors; original files kept in source-dir",
        source_rows=len(all_cases),
        exact_text_overlap_with_synthetic_development_ids=overlap_ids,
        sample_rows=len(cases),
        selection=f"first {args.sample_size} by SHA256('ko-pii-external-v1:' + document ID)",
        selected_ids=[c["id"] for c in cases],
        labels="PS -> KR_NAME; all other entity labels are non-person",
        limitation="Public/fictional people included; upstream model exposure unknown; "
        "single-source benchmark, not private-PII deployment accuracy. "
        "All extraction scores use public analyze; coverage is span-union, "
        "not a separate stars-mask run. "
        "Original article/thread groups unavailable; sentence bootstrap can understate dependence.",
        frozen_sha256=hashes,
        models={},
        runtime_promotion=False,
        guard_score_threshold=0.0,
        threshold_reason="Same public pipeline; research model threshold controls acceptance",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / (args.output.stem + "-manifest.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    ner = KoreanNER.from_pretrained(name_context_path=args.baseline)
    guard = KoreanPIIGuard(entities=["KR_NAME"], ner=ner, score_threshold=0.0)
    baseline = [
        {(r.start, r.end) for r in guard.analyze(c["text"]) if r.entity == "KR_NAME"} for c in cases
    ]
    report["baseline"] = detailed_metrics(cases, baseline)
    baseline_head = ner.name_context_head
    ner.name_context_head = None
    backbone_guard = KoreanPIIGuard(entities=["KR_NAME"], ner=ner, score_threshold=0.0)
    backbone_predictions = [
        {(r.start, r.end) for r in backbone_guard.analyze(case["text"]) if r.entity == "KR_NAME"}
        for case in cases
    ]
    report["backbone_e5_reference"] = dict(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        score_threshold=ner.score_threshold,
        metrics=detailed_metrics(cases, backbone_predictions),
        selection="Fixed reference, never selected or tuned on external scores",
    )
    ner.name_context_head = baseline_head
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
        candidate_guard = KoreanPIIGuard(entities=["KR_NAME"], ner=adapter, score_threshold=0.0)
        prediction = [
            {
                (r.start, r.end)
                for r in candidate_guard.analyze(case["text"])
                if r.entity == "KR_NAME"
            }
            for case in cases
        ]
        report["models"][directory.name] = dict(
            metrics=detailed_metrics(cases, prediction),
            regression_gate=gate(cases, baseline, prediction),
            paired_uncertainty=paired_f1_interval(cases, baseline, prediction),
            paired_vs_backbone=paired_f1_interval(cases, backbone_predictions, prediction),
        )
        print(
            directory.name,
            {k: v for k, v in report["models"][directory.name]["metrics"].items() if k != "errors"},
            flush=True,
        )
    f1_scores = [entry["metrics"]["f1"] for entry in report["models"].values()]
    report["seed_summary"] = dict(
        mean_f1=statistics.mean(f1_scores),
        minimum_f1=min(f1_scores),
        maximum_f1=max(f1_scores),
        sample_stdev_f1=statistics.stdev(f1_scores) if len(f1_scores) > 1 else None,
    )
    report["source_changed_during_run"] = any(digest(Path(p)) != h for p, h in hashes.items())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["source_changed_during_run"]:
        raise SystemExit("Frozen external evaluation inputs changed")


if __name__ == "__main__":
    main()
