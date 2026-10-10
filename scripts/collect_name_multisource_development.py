"""Cache the consumed KDPII v1 evaluation as next-generation development data.

This command cannot read unused test IDs. It requires a completed v1 report and
exactly matching previously exported cases. No candidate is promoted or selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ko_pii_guard.normalization import normalize_text


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_consumed(report, cases):
    if (report.get("scope") != "second_source_KDPII_frozen_candidate"
            or report.get("source_changed_during_run") is not False
            or not report.get("candidate", {}).get("metrics")):
        raise ValueError("Require a completed frozen, already consumed KDPII report")
    if (not cases or [c["id"] for c in cases] != report["selected_ids"]
            or len({c["id"] for c in cases}) != len(cases)):
        raise ValueError("Only the exact previously evaluated ID sequence may be opened")
    if any(normalize_text(c["text"]).text != c["text"] or not c["text"].strip()
           for c in cases):
        raise ValueError("Cache requires nonempty identity-normalized source coordinates")


def main():
    from name_adapted_adapter import build_ner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumed-report", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        parser.error("Use new paths; preserve earlier experiments")
    report = json.loads(args.consumed_report.read_text())
    cases = [json.loads(line) for line in args.cases.read_text().splitlines()]
    validate_consumed(report, cases)
    # The report pins the original exported cases, not merely their IDs.
    original = report["frozen_sha256"].get(str(args.cases.resolve()))
    if original != digest(args.cases):
        raise ValueError("Consumed source cases differ from the frozen evaluation")
    files = [Path(__file__), args.consumed_report, args.cases,
             *args.checkpoint.glob("*.json"), *args.checkpoint.glob("*.safetensors"),
             *[Path(__file__).with_name(name) for name in (
                 "name_adapted_adapter.py", "name_encoder_adaptation.py",
                 "name_generalization_model.py", "name_coverage_decoding.py",
                 "name_bioes_ablation.py")], *Path("src/ko_pii_guard").glob("*.py")]
    hashes = {str(path.resolve()): digest(path) for path in files}
    manifest = dict(scope="consumed_evaluation_now_development",
                    consumed_report=str(args.consumed_report),
                    selected_ids=[c["id"] for c in cases], frozen_sha256=hashes,
                    device="cpu", torch_version=str(torch.__version__),
                    checkpoint=str(args.checkpoint), independent_evaluation=False)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    torch.set_num_threads(4)
    ner = build_ner(dict(checkpoint=str(args.checkpoint), device="cpu",
                        threshold=0., decoder="viterbi"))
    logits, spans = [], []
    ner.on_logits = lambda value: logits.append(value.detach().cpu().clone())
    for index, case in enumerate(cases, 1):
        result = ner.analyze(case["text"])
        spans.append([dict(entity=r.entity_type, start=r.start, end=r.end, score=r.score)
                      for r in result])
        if index % 100 == 0 or index == len(cases):
            print(f"Consumed KDPII development {index}/{len(cases)}", flush=True)
    if len(logits) != len(cases):
        raise ValueError("Incomplete original-coordinate cache")
    if any(digest(path) != expected for path, expected in hashes.items()):
        raise ValueError("Inputs changed during collection")
    torch.save(dict(**manifest, logits=logits, spans=spans,
                    source_changed_during_run=False), args.output)


if __name__ == "__main__":
    main()
