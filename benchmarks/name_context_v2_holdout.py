"""Compare E5, v1 and v2 on the pre-frozen, separately named v2 evaluation split.

This is synthetic crossed-template validation, not independent real-world sample
accuracy. The evaluation is never used by training/selection in this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from name_address_benchmark import _dependency_versions, _source_metadata, evaluate, load_cases

from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard
from ko_pii_guard.name_context import load_name_head
from ko_pii_guard.ner import MODEL_ID, MODEL_REVISION, KoreanNER

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    data = ROOT / "benchmarks/data/name_context_v2.jsonl"
    cases = [c for c in load_cases(data) if c["split"] == "evaluation"]
    source, data_hash = _source_metadata(), digest(data)
    report = dict(
        scope="new split-separated SYNTHETIC evaluation; not field accuracy",
        data_sha256=data_hash,
        source=source,
        dependency_versions=_dependency_versions(),
        measured_at_utc=datetime.now(timezone.utc).isoformat(),
        benchmark_sha256=digest(Path(__file__)),
        profiles={},
        checkpoints={},
    )
    ner = KoreanNER.from_pretrained()
    guard = KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=ner)
    for profile in ("e5", "v1", "v2"):
        if profile != "e5":
            path = ROOT / f"artifacts/name-context-{profile}"
            ner.name_context_head = load_name_head(
                path, model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu"
            )
            report["checkpoints"][profile] = {
                name: digest(path / name)
                for name in ("name_context_config.json", "name_context.safetensors")
            }
        result = evaluate(cases, guard)
        report["profiles"][profile] = result
        print(profile, result["counts"], flush=True)
    report["source_changed_during_run"] = source != _source_metadata() or data_hash != digest(data)
    if report["source_changed_during_run"]:
        raise RuntimeError("Source or data changed during evaluation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
