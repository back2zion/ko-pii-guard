"""Compare frozen name recognizers through real analyze and stars-mask APIs.

The previous 1,000 KLUE sentences are development data. Preparing a split reads
only header IDs; held-out text/annotations are parsed only after a finalized
candidate selection and all frozen hashes have been checked.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import random
from pathlib import Path

from evaluate_name_span_external import parse_klue
from train_name_span_experiment import gate, gold, metrics

ROOT = Path(__file__).resolve().parents[1]
SPLIT_SALT = "ko-pii-generalization-v1:"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_ids(path):
    """Inspect headers only; do not parse or report any sentence/label."""
    with Path(path).open() as stream:
        ids = [line[3:].split("\t", 1)[0].strip() for line in stream
               if line.startswith("## klue-ner-")]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Empty corpus or duplicate header IDs")
    return ids


def make_split(ids, previous_ids, count):
    if len(set(ids)) != len(ids) or len(set(previous_ids)) != len(previous_ids):
        raise ValueError("Split contains duplicate IDs")
    if not set(previous_ids) <= set(ids):
        raise ValueError("Previous evaluation contains unknown IDs")
    available = set(ids) - set(previous_ids)
    if not 1 <= count <= len(available):
        raise ValueError("Invalid held-out sample size")
    selected = sorted(
        available, key=lambda identifier: hashlib.sha256(
            (SPLIT_SALT + identifier).encode()).hexdigest()
    )[:count]
    return dict(
        development_ids=sorted(previous_ids),
        heldout_ids=selected,
        untouched_remaining_rows=len(available) - count,
        selection=f"first {count} remaining IDs by SHA256('{SPLIT_SALT}' + ID)",
        preparation_reads="header IDs only; no sentence text or annotations parsed",
    )


def validate_split(split, ids, previous_ids):
    expected = make_split(ids, previous_ids, len(split["heldout_ids"]))
    if any(split.get(key) != value for key, value in expected.items()):
        raise ValueError("Split differs from predeclared header-only selection")


def read_selected_cases(path, selected_ids):
    """Parse only the selected records, preserving original offsets."""
    wanted, lines, active = set(selected_ids), [], False
    if len(wanted) != len(selected_ids):
        raise ValueError("Duplicate selected IDs")
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith("## klue-ner-"):
                active = line[3:].split("\t", 1)[0].strip() in wanted
            if active:
                lines.append(line)
    selected = {case["id"]: case for case in parse_klue("".join(lines))}
    if set(selected) != wanted:
        raise ValueError("Selected IDs are absent from source")
    return [selected[identifier] for identifier in selected_ids]


def validate_selection(selection, *, heldout):
    if heldout and selection.get("finalized") is not True:
        raise ValueError("Held-out evaluation requires finalized candidate selection")
    if not selection.get("selection_basis") or not isinstance(selection.get("candidate"), dict):
        raise ValueError("Selection must describe development selection and candidate config")
    hashes = selection.get("frozen_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Selection requires nonempty frozen_sha256 inputs")
    for filename, expected in hashes.items():
        if digest(filename) != expected:
            raise ValueError(f"Selected candidate input changed: {filename}")
    load_training_text_hashes(selection, heldout=heldout)


def load_training_text_hashes(selection, *, heldout):
    path = selection.get("training_text_hashes_json")
    if path is None:
        if heldout and selection["candidate"].get("mode", "learned") != "e5":
            raise ValueError("Learned held-out candidate requires training_text_hashes_json")
        return None
    frozen = {str(Path(p).resolve()): h for p, h in selection["frozen_sha256"].items()}
    if frozen.get(str(Path(path).resolve())) != digest(path):
        raise ValueError("Training text hashes must be an unchanged frozen input")
    data = json.loads(Path(path).read_text())
    hashes = data.get("text_sha256")
    if (data.get("algorithm") != "sha256_utf8_exact_text" or not isinstance(hashes, list)
            or any(not isinstance(h, str) or len(h) != 64
                   or any(char not in "0123456789abcdef" for char in h) for h in hashes)):
        raise ValueError("Expected SHA256 hashes of exact UTF-8 training/validation text")
    return set(hashes)


def training_overlap_ids(cases, training_hashes):
    if training_hashes is None:
        return None
    return [case["id"] for case in cases
            if hashlib.sha256(case["text"].encode()).hexdigest() in training_hashes]


def validate_devices(baseline, candidate):
    import torch

    def resolved(ner):
        device = torch.device(getattr(ner, "backbone", ner).device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device()
                                  if torch.cuda.is_available() else 0)
        return str(device)

    devices = {"baseline": resolved(baseline), "candidate": resolved(candidate)}
    if devices["baseline"] != devices["candidate"]:
        raise ValueError(f"Baseline and candidate devices differ: {devices}")
    return devices


def _score(cases, predictions, masks):
    result = metrics(cases, predictions)
    expected_count = sum(len(gold(case)) for case in cases)
    result["precision"] = result["tp"] / (result["tp"] + result["fp"]) if (
        result["tp"] + result["fp"]) else 0.0
    result["recall"] = result["tp"] / expected_count if expected_count else 0.0
    result["fully_covered_names"] = sum(
        masked[start:end] == "*" * (end - start)
        for case, masked in zip(cases, masks, strict=True) for start, end in gold(case)
    )
    result["full_name_coverage"] = (
        result["fully_covered_names"] / expected_count if expected_count else 0.0
    )
    result["unnecessary_masked_characters"] = sum(
        char != "*" and masked[i] == "*" and i not in {
            index for start, end in gold(case) for index in range(start, end)
        }
        for case, masked in zip(cases, masks, strict=True)
        for i, char in enumerate(case["text"])
    )
    result["negative_sentences"] = sum(not gold(case) for case in cases)
    result["negative_false_positive_sentences"] = sum(
        not gold(case) and bool(predicted)
        for case, predicted in zip(cases, predictions, strict=True)
    )
    result["sentences"] = len(cases)
    result["gold_names"] = expected_count
    return result


def evaluate_guard(cases, guard, *, progress=None):
    predictions, masks = [], []
    for index, case in enumerate(cases, 1):
        predictions.append({(r.start, r.end) for r in guard.analyze(case["text"])
                            if r.entity == "KR_NAME"})
        masked = guard.mask(case["text"], style="stars")
        if len(masked) != len(case["text"]):
            raise ValueError("Stars mask changed original coordinate system")
        masks.append(masked)
        if progress and (index % 100 == 0 or index == len(cases)):
            progress(index, len(cases))
    result = _score(cases, predictions, masks)
    result["by_source"] = {}
    for source in sorted({case["id"].rsplit("-", 1)[-1] for case in cases}):
        positions = [i for i, case in enumerate(cases)
                     if case["id"].rsplit("-", 1)[-1] == source]
        source_score = _score([cases[i] for i in positions],
                              [predictions[i] for i in positions],
                              [masks[i] for i in positions])
        result["by_source"][source] = {k: v for k, v in source_score.items() if k != "errors"}
    return result, predictions, masks


def actual_mask_gate(cases, baseline, predictions, baseline_masks, masks):
    regression = gate(cases, baseline, predictions)
    regression["actual_stars_newly_exposed_names"] = [
        (case["id"], start, end)
        for case, old, new in zip(cases, baseline_masks, masks, strict=True)
        for start, end in sorted(gold(case))
        if old[start:end] == "*" * (end - start) and new[start:end] != "*" * (end - start)
    ]
    regression["passed"] &= not regression["actual_stars_newly_exposed_names"]
    return regression


def paired_intervals(cases, baseline, predictions, baseline_masks, masks, *, repetitions=500):
    """Paired sentence bootstrap; document/thread grouping is unavailable."""
    if not cases or repetitions < 2:
        raise ValueError("Bootstrap needs sentences and at least two repetitions")
    rows = []
    for case, old, new, old_mask, new_mask in zip(
        cases, baseline, predictions, baseline_masks, masks, strict=True
    ):
        expected = gold(case)
        rows.append(tuple(value for spans, masked in ((old, old_mask), (new, new_mask))
                          for value in (len(expected & spans), len(spans - expected),
                                        len(expected - spans), sum(
                                            masked[s:e] == "*" * (e - s) for s, e in expected))))

    def ratios(tp, fp, fn, covered):
        return (2 * tp / (2 * tp + fp + fn) if tp else 0.0,
                tp / (tp + fp) if tp else 0.0,
                tp / (tp + fn) if tp else 0.0,
                covered / (tp + fn) if tp + fn else 0.0)

    rng = random.Random(20261010)
    deltas = {key: [] for key in ("f1", "precision", "recall", "full_name_coverage")}
    for _ in range(repetitions):
        total = [0] * 8
        for _ in cases:
            for i, value in enumerate(rows[rng.randrange(len(rows))]):
                total[i] += value
        old, new = ratios(*total[:4]), ratios(*total[4:])
        for key, a, b in zip(deltas, old, new, strict=True):
            deltas[key].append(b - a)
    intervals = {}
    for key, values in deltas.items():
        values.sort()
        intervals[key] = [values[int(0.025 * (repetitions - 1))],
                          values[int(0.975 * (repetitions - 1))]]
    return dict(method="paired sentence percentile bootstrap", repetitions=repetitions,
                seed=20261010, delta_95_intervals=intervals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "development", "heldout"), required=True)
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/ko-pii-klue-ner"))
    parser.add_argument("--previous-evaluation", type=Path,
                        default=ROOT / "benchmarks/results/name-span-v1-external.json")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--heldout-size", type=int, default=1000)
    parser.add_argument("--candidate-factory", help="MODULE:FUNCTION; receives candidate config")
    parser.add_argument("--selection", type=Path, help="Frozen candidate selection JSON")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    source_path = args.source_dir / "klue-ner-v1.1_dev.tsv"
    source = json.loads((args.source_dir / "source.json").read_text())
    for filename, metadata in source["files"].items():
        if digest(args.source_dir / filename) != metadata["sha256"]:
            raise ValueError("Official external source changed")
    if args.mode == "prepare":
        if args.split_manifest.exists():
            parser.error("Do not replace a predeclared split manifest")
        previous = json.loads(args.previous_evaluation.read_text())
        split = make_split(collect_ids(source_path), previous["selected_ids"], args.heldout_size)
        split.update(source=source, source_sha256=digest(source_path),
                     previous_evaluation_sha256=digest(args.previous_evaluation))
        args.split_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.split_manifest.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n")
        print(f"Frozen {len(split['heldout_ids'])} held-out IDs; no held-out labels parsed")
        return
    if not args.candidate_factory or not args.selection or not args.output:
        parser.error("Evaluation requires --candidate-factory, --selection, and --output")
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        parser.error("Use new output paths; evaluations cannot be overwritten")
    selection = json.loads(args.selection.read_text())
    validate_selection(selection, heldout=args.mode == "heldout")
    split = json.loads(args.split_manifest.read_text())
    if split["source_sha256"] != digest(source_path):
        raise ValueError("Split manifest source changed")
    if split["previous_evaluation_sha256"] != digest(args.previous_evaluation):
        raise ValueError("Previously inspected evaluation changed")
    previous = json.loads(args.previous_evaluation.read_text())
    validate_split(split, collect_ids(source_path), previous["selected_ids"])
    factory_module, factory_name = args.candidate_factory.rsplit(":", 1)
    module = importlib.import_module(factory_module)
    factory = getattr(module, factory_name)
    files = [Path(__file__), Path(module.__file__), args.selection, args.split_manifest,
             source_path, args.source_dir / "source.json", args.previous_evaluation,
             ROOT / "scripts/evaluate_name_span_external.py",
             ROOT / "scripts/train_name_span_experiment.py",
             *sorted((ROOT / "src/ko_pii_guard").glob("*.py"))]
    hashes = {**selection["frozen_sha256"], **{str(p.resolve()): digest(p) for p in files}}
    selected_ids = split["development_ids" if args.mode == "development" else "heldout_ids"]
    report = dict(
        scope=args.mode, selected_ids=selected_ids, selection=selection,
        candidate_factory=args.candidate_factory, frozen_sha256=hashes, source=source,
        guard_score_threshold=0.0, ner_baseline_score_threshold=0.9,
        pipeline="Public KoreanPIIGuard.analyze and separate actual mask(style='stars') calls",
        runtime_promotion=False,
        limitations="KLUE PS boundary stress test, including public/fictional people; not "
        "private-person policy accuracy. Upstream model exposure unknown. Previous 1,000 "
        "sentences are development. Bootstrap unit is sentence; original source-document "
        "grouping unavailable. Candidate selection must not use held-out results.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    # Selection, factory, and hashes are frozen before opening selected annotations.
    cases = read_selected_cases(source_path, selected_ids)
    overlap = training_overlap_ids(
        cases, load_training_text_hashes(selection, heldout=args.mode == "heldout"))
    report["exact_training_or_validation_text_overlap_ids"] = overlap
    if args.mode == "heldout" and overlap:
        report["evaluation_blocked"] = "heldout_training_or_validation_text_overlap"
        report["source_changed_during_run"] = any(digest(p) != h for p, h in hashes.items())
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        raise SystemExit(f"Held-out text overlap detected for IDs {overlap}; no resampling")
    import torch
    import transformers

    from ko_pii_guard import KoreanPIIGuard
    from ko_pii_guard.ner import MODEL_ID, MODEL_REVISION, KoreanNER

    torch.set_num_threads(2)
    baseline_ner = KoreanNER.from_pretrained(device=args.device)
    candidate_ner = factory(selection["candidate"])
    try:
        devices = validate_devices(baseline_ner, candidate_ner)
    except ValueError as error:
        report["evaluation_blocked"] = str(error)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        raise
    report["runtime"] = dict(python=platform.python_version(), torch=torch.__version__,
                             transformers=transformers.__version__, devices=devices)
    baseline_guard = KoreanPIIGuard(entities=["KR_NAME"], ner=baseline_ner, score_threshold=0.0)
    baseline_score, baseline, baseline_masks = evaluate_guard(
        cases, baseline_guard, progress=lambda i, n: print(f"baseline {i}/{n}", flush=True))
    report["baseline"] = dict(model_id=MODEL_ID, revision=MODEL_REVISION, metrics=baseline_score)
    candidate_guard = KoreanPIIGuard(entities=["KR_NAME"], ner=candidate_ner, score_threshold=0.0)
    score, predictions, masks = evaluate_guard(
        cases, candidate_guard, progress=lambda i, n: print(f"candidate {i}/{n}", flush=True))
    report["candidate"] = dict(
        metrics=score,
        regression_gate=actual_mask_gate(cases, baseline, predictions, baseline_masks, masks),
        bootstrap=paired_intervals(cases, baseline, predictions, baseline_masks, masks),
    )
    report["source_changed_during_run"] = any(digest(p) != h for p, h in hashes.items())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({name: {k: v for k, v in report[name]["metrics"].items()
                            if k not in ("errors", "by_source")}
                      for name in ("baseline", "candidate")}, ensure_ascii=False), flush=True)
    if report["source_changed_during_run"]:
        raise SystemExit("Frozen inputs changed during evaluation; results cannot be accepted")


if __name__ == "__main__":
    main()
