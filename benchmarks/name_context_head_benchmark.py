"""Compare the opt-in trained head on existing development regression corpora.

The frozen training run has separate held-out results; these corpora are reused
engineering checks and must never be reported as new held-out evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from name_address_benchmark import _dependency_versions, _source_metadata, evaluate, load_cases

from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard
from ko_pii_guard.ner import KoreanNER


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    data = Path(__file__).parent / "data"
    filenames = ("name_address.jsonl", "business_korean.jsonl", "name_field_boundaries.jsonl")
    source = _source_metadata()
    report = dict(
        scope="existing synthetic DEVELOPMENT regressions; not held-out accuracy",
        measured_at_utc=datetime.now(timezone.utc).isoformat(),
        source=source,
        dependencies=_dependency_versions(),
        checkpoint_sha256={
            name: hashlib.sha256((args.checkpoint / name).read_bytes()).hexdigest()
            for name in ("name_context_config.json", "name_context.safetensors")
        },
        data_sha256={
            name: hashlib.sha256((data / name).read_bytes()).hexdigest() for name in filenames
        },
        profiles={},
    )
    ner = KoreanNER.from_pretrained()
    guard = KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=ner)
    for label in ("baseline", "candidate"):
        if label == "candidate":
            from ko_pii_guard.name_context import load_name_head
            from ko_pii_guard.ner import MODEL_ID, MODEL_REVISION

            ner.name_context_head = load_name_head(
                args.checkpoint, model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu"
            )
        report["profiles"][label] = {}
        for name in filenames:
            result = evaluate(load_cases(data / name), guard)
            report["profiles"][label][name] = result
            print(label, name, result["counts"], flush=True)
    report["source_changed_during_run"] = source != _source_metadata() or any(
        hashlib.sha256((data / name).read_bytes()).hexdigest() != report["data_sha256"][name]
        for name in filenames
    )
    if report["source_changed_during_run"]:
        raise RuntimeError("Runtime source changed during evaluation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
