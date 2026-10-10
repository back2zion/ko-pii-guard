"""Public API acceptance for retired development data or frozen new v4 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from name_address_benchmark import _source_metadata, evaluate, load_cases

from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard
from ko_pii_guard.ner import KoreanNER

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/data"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "evaluation"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/name-context-v4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check and args.split != "development":
        parser.error("--check is a development acceptance gate, not a held-out selection gate")
    torch.set_num_threads(2)
    if args.split == "development":
        corpora = {
            "v3_retired_evaluation": load_cases(DATA / "name_context_v3_regressions.jsonl"),
            "v2_retired_evaluation": load_cases(DATA / "name_context_v2_regressions.jsonl"),
            "v1_retired_evaluation": [c for c in load_cases(
                DATA / "name_context_training.jsonl"
            ) if c["split"] == "evaluation"],
        }
        corpora.update({name: load_cases(DATA / name) for name in (
            "name_context.jsonl", "name_address.jsonl", "business_korean.jsonl",
            "name_field_boundaries.jsonl",
        )})
    else:
        corpora = {"v4_new_evaluation": [c for c in load_cases(
            DATA / "name_context_v4.jsonl"
        ) if c["split"] == "evaluation"]}
    data_hashes = {p.name: digest(p) for p in DATA.glob("*.jsonl")}
    checkpoint_hashes = {name: digest(args.checkpoint / name) for name in (
        "name_context_config.json", "name_context.safetensors"
    )}
    sources = _source_metadata()
    guard = KoreanPIIGuard(
        entities=SUPPORTED_ENTITIES,
        ner=KoreanNER.from_pretrained(name_context_path=args.checkpoint),
    )
    results = {}
    for name, cases in corpora.items():
        results[name] = evaluate(cases, guard)
        print(name, results[name]["counts"], flush=True)
    report = dict(
        scope=("KNOWN DEVELOPMENT acceptance" if args.split == "development" else
               "NEW split-separated SYNTHETIC evaluation; not real-world accuracy"),
        measured_at_utc=datetime.now(timezone.utc).isoformat(),
        benchmark_sha256=digest(Path(__file__)), source=sources, data_sha256=data_hashes,
        checkpoint_sha256=checkpoint_hashes,
        results=results,
    )
    report["acceptance_passed"] = all(
        r["counts"]["false_positive"] == r["counts"]["false_negative"] == 0
        for r in results.values()
    ) if args.split == "development" else None
    report["source_changed_during_run"] = sources != _source_metadata() or any(
        digest(DATA / name) != value for name, value in data_hashes.items()
    ) or any(
        digest(args.checkpoint / name) != value for name, value in checkpoint_hashes.items()
    )
    if report["source_changed_during_run"]:
        raise RuntimeError("Source/data changed during measurement")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if args.check and not report["acceptance_passed"]:
        raise SystemExit("Known development errors/regressions remain")


if __name__ == "__main__":
    main()
