"""Measure fixed adapted-name checkpoint CPU/CUDA differences on old development.

The sample is declared from already inspected development IDs before its text
is opened. This is a deployment diagnostic, never a checkpoint-selection metric.
Only coordinates and scores are saved; original sentences remain in source-dir.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import statistics
import time
from pathlib import Path

import torch
from evaluate_name_generalization import _score, actual_mask_gate, read_selected_cases
from name_adapted_adapter import build_ner

from ko_pii_guard import KoreanPIIGuard

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_ids(development, count):
    if (development.get("scope") != "development"
            or development.get("source_changed_during_run") is not False):
        raise ValueError("Only completed, unchanged previous development is allowed")
    identifiers = development.get("selected_ids", [])
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Development must contain unique sentence IDs")
    if not 1 <= count <= 100:
        raise ValueError("Device parity sample must contain at most 100 sentences")
    return sorted(identifiers)[:count]


def compare_outputs(cases, cpu, cuda):
    disagreements = []
    counts = dict(span_sentences=0, score_sentences=0, mask_sentences=0, mask_characters=0)
    for index, case in enumerate(cases):
        cpu_spans, cuda_spans = cpu["predictions"][index], cuda["predictions"][index]
        cpu_scores, cuda_scores = cpu["scores"][index], cuda["scores"][index]
        old, new = cpu["masks"][index], cuda["masks"][index]
        if len(old) != len(case["text"]) or len(new) != len(case["text"]):
            raise ValueError("Actual stars masks must preserve original coordinates")
        positions = [i for i, (a, b) in enumerate(zip(old, new, strict=True)) if a != b]
        common = cpu_spans & cuda_spans
        score_changes = [dict(span=list(span), cpu=cpu_scores[span], cuda=cuda_scores[span])
                         for span in sorted(common) if cpu_scores[span] != cuda_scores[span]]
        counts["span_sentences"] += cpu_spans != cuda_spans
        counts["score_sentences"] += bool(score_changes)
        counts["mask_sentences"] += bool(positions)
        counts["mask_characters"] += len(positions)
        if cpu_spans != cuda_spans or score_changes or positions:
            disagreements.append(dict(id=case["id"], cpu_spans=sorted(cpu_spans),
                                      cuda_spans=sorted(cuda_spans), score_changes=score_changes,
                                      differing_mask_positions=positions))
    return dict(
        counts=counts, disagreements=disagreements,
        cpu_to_cuda_regression_gate=actual_mask_gate(
            cases, cpu["predictions"], cuda["predictions"], cpu["masks"], cuda["masks"]),
        cuda_to_cpu_regression_gate=actual_mask_gate(
            cases, cuda["predictions"], cpu["predictions"], cuda["masks"], cpu["masks"]),
    )


def run_device(cases, config, device, *, factory=build_ner):
    ner = factory({**config, "device": device})
    guard = KoreanPIIGuard(entities=["KR_NAME"], ner=ner, score_threshold=0.)
    output = dict(predictions=[], masks=[], scores=[])
    durations = []
    started = time.monotonic()
    for index, case in enumerate(cases, 1):
        before = time.monotonic()
        findings = [finding for finding in guard.analyze(case["text"])
                    if finding.entity == "KR_NAME"]
        output["predictions"].append({(finding.start, finding.end) for finding in findings})
        output["scores"].append({(finding.start, finding.end): finding.score
                                 for finding in findings})
        masked = guard.mask(case["text"], style="stars")
        if len(masked) != len(case["text"]):
            raise ValueError("Actual stars mask changed original text coordinates")
        output["masks"].append(masked)
        durations.append(time.monotonic() - before)
        if index % 10 == 0 or index == len(cases):
            print(f"{device}: {index}/{len(cases)} actual analyze + stars "
                  f"({time.monotonic() - started:.1f}s)", flush=True)
    ordered = sorted(durations)
    output["metrics"] = _score(cases, output["predictions"], output["masks"])
    output["execution"] = dict(
        device=device, decoder=ner.decoder_name, threshold=ner.threshold,
        adapted_encoder_precision=(ner.encoder_autocast if device == "cuda" else "float32"),
        original_e5_prior_precision="float32", elapsed_seconds=time.monotonic() - started,
        analyze_plus_stars_median_seconds=statistics.median(durations),
        analyze_plus_stars_p95_seconds=ordered[int(0.95 * (len(ordered) - 1))],
        timing_scope="Model loading excluded; two API calls per sentence; includes warmup",
    )
    del guard, ner
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--candidate-factory", default="name_adapted_adapter:build_ner",
                        help="MODULE:FUNCTION accepting a fixed candidate configuration")
    parser.add_argument("--decoder", choices=("viterbi", "coverage"))
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--source", type=Path,
                        default=Path("/tmp/ko-pii-klue-ner/klue-ner-v1.1_dev.tsv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        parser.error("Use new output paths; parity evidence must not be overwritten")
    if args.selection and (args.decoder is not None or args.threshold is not None):
        parser.error("A fixed selection cannot be overridden with decoder or threshold")
    if not torch.cuda.is_available():
        parser.error("This comparison requires both CPU and accessible CUDA")
    development = json.loads(args.development.read_text())
    identifiers = sample_ids(development, args.sample_size)
    if digest(args.source) != development["source"]["files"][args.source.name]["sha256"]:
        raise ValueError("Previous development source bytes changed")
    training = json.loads((args.checkpoint / "report.json").read_text())
    if training.get("source_changed") is not False:
        raise ValueError("Only a completed, unchanged trained checkpoint may be compared")
    checkpoint = json.loads((args.checkpoint / "config.json").read_text())
    config = dict(checkpoint=str(args.checkpoint), decoder=checkpoint.get("decoder", "viterbi"),
                  threshold=checkpoint["threshold"])
    if args.selection:
        selection = json.loads(args.selection.read_text())
        selected = selection["candidate"]
        location = selected.get("checkpoint", selected.get("path"))
        if location is None or Path(location).resolve() != args.checkpoint.resolve():
            raise ValueError("Selection and checkpoint must identify the same weights")
        for filename, expected in selection.get("frozen_sha256", {}).items():
            if digest(filename) != expected:
                raise ValueError("Previously selected input changed")
        config.update({key: selected[key] for key in ("decoder", "threshold") if key in selected})
    else:
        if args.decoder is not None:
            config["decoder"] = args.decoder
        if args.threshold is not None:
            config["threshold"] = args.threshold
    module_name, factory_name = args.candidate_factory.rsplit(":", 1)
    factory_module = importlib.import_module(module_name)
    factory = getattr(factory_module, factory_name)
    if not callable(factory):
        raise ValueError("Candidate factory must be callable")
    files = [Path(__file__), Path(factory_module.__file__), args.source, args.development,
             *args.checkpoint.glob("*.json"), *args.checkpoint.glob("*.safetensors"),
             *[ROOT / "scripts" / name for name in (
                 "name_adapted_adapter.py", "name_encoder_adaptation.py",
                 "name_generalization_model.py", "name_coverage_decoding.py",
                 "name_bioes_ablation.py", "evaluate_name_generalization.py",
                 "evaluate_name_span_external.py", "train_name_span_experiment.py")],
             *sorted((ROOT / "src/ko_pii_guard").glob("*.py"))]
    if args.selection:
        files.append(args.selection)
    hashes = {str(path.resolve()): digest(path) for path in files}
    report = dict(
        scope="development_device_parity", selected_ids=identifiers,
        selection="First at most 100 previously inspected development IDs in lexical order",
        candidate_config=config, frozen_sha256=hashes,
        candidate_factory=args.candidate_factory,
        pipeline="Separate actual public analyze and mask(style='stars') calls on each device",
        guard_score_threshold=0., score_precision="Public API confidence rounded to three decimals",
        model_selection_from_results=False, runtime_promotion=False,
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        cuda_device=torch.cuda.get_device_name(0),
        limitations="Bounded development diagnostic; no independent accuracy claim. CPU adapted "
                    "encoder uses FP32, CUDA uses checkpoint-configured precision. A zero "
                    "disagreement result on this sample does not prove universal device parity.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    # Freeze sample IDs, configuration, and sources before opening their annotations.
    cases = read_selected_cases(args.source, identifiers)
    torch.set_num_threads(2)
    cpu = run_device(cases, config, "cpu", factory=factory)
    cuda = run_device(cases, config, "cuda", factory=factory)
    report["devices"] = {device: {key: result[key] for key in ("metrics", "execution")}
                         for device, result in (("cpu", cpu), ("cuda", cuda))}
    report["comparison"] = compare_outputs(cases, cpu, cuda)
    report["source_changed_during_run"] = any(digest(path) != expected
                                              for path, expected in hashes.items())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["comparison"]["counts"]), flush=True)
    if report["source_changed_during_run"]:
        raise ValueError("Frozen device-parity inputs changed during evaluation")


if __name__ == "__main__":
    main()
