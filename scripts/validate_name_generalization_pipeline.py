"""Finish the fixed development comparison before opening either fresh source.

Run from the repository root in the isolated CUDA environment. Complete the
fixed three-epoch adaptation if its finished checkpoint is not present yet.
No parameter or model selection uses held-out, KDPII, or device-parity results.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from select_name_coverage import selection_rank

RESULTS = Path("benchmarks/results")
DEVELOPMENT = RESULTS / "name-generalization-v1-development.json"
ADAPTED = Path("artifacts/name-generalization-lora-v1")
FROZEN = Path("artifacts/name-generalization-v1")
PREFIX = "name-generalization-final-v1"


def run(script, *args):
    command = [sys.executable, "-u", f"scripts/{script}", *map(str, args)]
    print("Running", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def read(path):
    return json.loads(path.read_text())


def continue_after_overlap(blocked_path):
    """Retain the fixed candidate and exclude contamination without resampling.

The original 1,000-row run remains blocked and preserved. This amendment is
declared before any held-out inference and changes neither weights nor policy.
"""
    blocked = read(blocked_path)
    if (blocked.get("evaluation_blocked") != "heldout_training_or_validation_text_overlap"
            or blocked.get("source_changed_during_run") is not False):
        raise ValueError("Only the preserved pre-inference contamination block can resume")
    old_path = RESULTS / f"{PREFIX}-selection.json"
    old = read(old_path)
    archive = ADAPTED / "validate_name_generalization_pipeline.py.source.txt"
    source = Path(__file__).resolve()
    hashes = {}
    for filename, expected in old["frozen_sha256"].items():
        path = archive if Path(filename).resolve() == source else Path(filename)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Original fixed candidate or provenance changed: {filename}")
        hashes[str(path)] = expected
    for path in (source, old_path, blocked_path,
                 Path("scripts/evaluate_name_generalization_nonoverlap.py")):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    selection = dict(old, frozen_sha256=hashes, amendment=
                     "Original 1000-ID evaluation stopped before inference on exact training "
                     "text overlap. Keep identical model/decoder/threshold and original IDs; "
                     "exclude only exact overlap, without replacement. This is not retuning.")
    selection_path = RESULTS / f"{PREFIX}-nonoverlap-selection.json"
    if selection_path.exists():
        raise ValueError("Amended selection exists; do not overwrite evaluation evidence")
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
    print("Continuing unchanged candidate after contamination exclusion", flush=True)
    factory = selection["candidate_factory"]
    common = ("--candidate-factory", factory, "--selection", selection_path, "--device", "cuda")

    def output(suffix):
        return RESULTS / f"{PREFIX}-nonoverlap-{suffix}.json"

    run("evaluate_name_generalization.py", "--mode", "development", "--split-manifest",
        RESULTS / "name-generalization-v1-split.json", *common,
        "--output", output("development"))
    previous = read(RESULTS / f"{PREFIX}-development.json")
    validate_live_development(read(output("development")),
                              previous["candidate"]["metrics"], previous["baseline"]["metrics"])
    run("evaluate_name_generalization_nonoverlap.py", "--split-manifest",
        RESULTS / "name-generalization-v1-split.json", "--blocked-report", blocked_path,
        *common, "--output", output("heldout"))
    run("evaluate_name_kdpii.py", *common, "--output", output("kdpii"))
    run("check_name_adaptation_device_parity.py", "--checkpoint", selection["candidate"][
        "checkpoint"], "--candidate-factory", factory, "--development", DEVELOPMENT,
        "--selection", selection_path, "--output", output("device-parity"))
    print("Non-overlapping fixed-candidate evaluations complete.", flush=True)


def _verify_hashes(hashes, label):
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"{label} requires nonempty frozen hashes")
    for filename, expected in hashes.items():
        path = Path(filename)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{label} missing or changed: {filename}")


def verify_policy_provenance(policy):
    """Validate current policy inputs and the cached encoder's transitive inputs."""
    if policy.get("source_changed_during_run") is not False:
        raise ValueError("Development selection source changed")
    hashes = policy.get("source_sha256")
    _verify_hashes(hashes, "Policy input")
    caches = [Path(filename) for filename in hashes if Path(filename).suffix == ".pt"]
    if len(caches) != 1:
        raise ValueError("Policy must identify exactly one frozen development .pt cache")
    import torch

    cache = torch.load(caches[0], map_location="cpu", weights_only=True)
    if not isinstance(cache, dict):
        raise ValueError("Cached development input requires a metadata dictionary")
    _verify_hashes(cache.get("frozen_sha256"), "Cached development input")


def validate_live_development(live, expected_score, expected_baseline):
    """Require per-case replay equality and unchanged feasible selection baseline."""
    if live.get("scope") != "development" or live.get("source_changed_during_run") is not False:
        raise ValueError("Require completed live development with unchanged sources")
    actual = live["candidate"]["metrics"]
    baseline = live["baseline"]["metrics"]
    # Metrics include every error's document ID and coordinates, not just totals.
    if actual != expected_score:
        raise ValueError("Live candidate differs from replay; fresh data stays closed")
    if baseline != expected_baseline:
        raise ValueError("Live baseline differs from selection baseline; fresh data stays closed")
    if selection_rank(actual, baseline) is None:
        raise ValueError("Live candidate fails selection criteria; fresh data stays closed")


def main():
    blocked_path = RESULTS / f"{PREFIX}-heldout.json"
    if blocked_path.exists():
        continue_after_overlap(blocked_path)
        return
    data = Path("/tmp/ko-pii-name-generalization-v1-verified")
    if not (ADAPTED / "report.json").exists():
        run("train_name_encoder_adaptation.py", "--data-dir", data,
            "--initial", FROZEN, "--cache", "/tmp/name-generalization-v1-features.pt",
            "--output", ADAPTED, "--device", "cuda")
    if read(ADAPTED / "report.json").get("source_changed") is not False:
        raise ValueError("Adaptation training must finish with unchanged sources")
    for source, expected in read(ADAPTED / "manifest.json")["source_sha256"].items():
        path = Path(source)
        if path.suffix == ".py":
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Executed training source changed: {path}")
            (ADAPTED / f"{path.name}.source.txt").write_bytes(content)
    (ADAPTED / "data-manifest.json").write_bytes((data / "manifest.json").read_bytes())
    (ADAPTED / "training-text-hashes.json").write_bytes(
        (FROZEN / "training-text-hashes.json").read_bytes())
    def output(suffix):
        return RESULTS / f"{PREFIX}-{suffix}.json"

    selection_path = output("selection")
    if any(output(s).exists() for s in ("selection", "development", "heldout", "kdpii",
                                       "device-parity")):
        raise ValueError("Final outputs already exist; do not overwrite or retune")
    cache = Path("/tmp/name-generalization-lora-v1-development-logits.pt")
    spans = RESULTS / "name-generalization-lora-v1-development.spans.jsonl"
    diagnostic = RESULTS / "name-generalization-lora-v1-development-diagnostic.json"
    adapted_policy = RESULTS / "name-generalization-lora-v1-coverage-selection.json"
    run("collect_adapted_name_development.py", "--checkpoint", ADAPTED,
        "--development", DEVELOPMENT, "--cache", cache, "--spans", spans,
        "--output", diagnostic, "--device", "cuda")
    run("select_name_coverage.py", "--development", DEVELOPMENT, "--cache", cache,
        "--spans", spans, "--output", adapted_policy)

    baseline = read(DEVELOPMENT)["baseline"]["metrics"]
    policies = [
        (FROZEN, "name_adapted_adapter:build_frozen_ner",
         RESULTS / "name-generalization-v1-coverage-selection.json"),
        (ADAPTED, "name_adapted_adapter:build_ner", adapted_policy),
    ]
    best = None
    for checkpoint, factory, policy_path in policies:
        policy = read(policy_path)
        verify_policy_provenance(policy)
        cutoff = policy["selected"]
        if cutoff is None:
            continue
        score = policy["results"][str(cutoff)]["metrics"]
        rank = selection_rank(score, baseline)
        if rank is not None and (best is None or rank > best[0]):
            best = rank, checkpoint, factory, cutoff, score
    if best is None:
        raise ValueError("No candidate satisfies the predeclared development criteria")
    _, checkpoint, factory, cutoff, expected_score = best
    files = [*checkpoint.glob("*.safetensors"), checkpoint / "config.json",
             checkpoint / "report.json", FROZEN / "training-text-hashes.json",
             DEVELOPMENT, *[row[2] for row in policies], Path(__file__),
             *[Path("scripts") / filename for filename in (
                 "name_adapted_adapter.py", "name_encoder_adaptation.py",
                 "name_generalization_model.py", "name_coverage_decoding.py",
                 "name_bioes_ablation.py", "select_name_coverage.py")]]
    selection = dict(
        finalized=True,
        selection_basis="Compare frozen and adapted encoders only on prior development. "
        "Fixed posterior cutoffs; full coverage/F1 no lower and negative FP/extra characters "
        "no higher than original E5. Maximize covered names, minimize extra characters, then "
        "maximize F1; retain frozen incumbent on a tie. Strict per-case regressions remain "
        "separate and cannot be waived by this ranking.",
        candidate_factory=factory,
        candidate=dict(checkpoint=str(checkpoint), device="cuda", decoder="coverage",
                       threshold=cutoff),
        training_text_hashes_json=str(FROZEN / "training-text-hashes.json"),
        frozen_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    )
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
    print("Final frozen candidate", selection["candidate"], flush=True)
    common = ("--candidate-factory", factory, "--selection", selection_path, "--device", "cuda")
    for mode in ("development", "heldout"):
        run("evaluate_name_generalization.py", "--mode", mode, "--split-manifest",
            RESULTS / "name-generalization-v1-split.json", *common, "--output", output(mode))
        if mode == "development":
            validate_live_development(read(output(mode)), expected_score, baseline)
    run("evaluate_name_kdpii.py", *common, "--output", output("kdpii"))
    run("check_name_adaptation_device_parity.py", "--checkpoint", checkpoint,
        "--candidate-factory", factory, "--development", DEVELOPMENT,
        "--selection", selection_path, "--output", output("device-parity"))
    print("Fixed-candidate evaluations complete; inspect regression gates before adoption.",
          flush=True)


if __name__ == "__main__":
    main()
