"""Cache a selected adapted encoder on previously inspected development only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from evaluate_name_generalization import read_selected_cases
from name_adapted_adapter import build_ner
from train_name_span_experiment import metrics

from ko_pii_guard import KoreanPIIGuard
from ko_pii_guard.normalization import normalize_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source", type=Path,
                        default=Path("/tmp/ko-pii-klue-ner/klue-ner-v1.1_dev.tsv"))
    args = parser.parse_args()
    if any(p.exists() for p in (args.cache, args.spans, args.output)):
        parser.error("New paths required; preserve prior experiment results")
    development = json.loads(args.development.read_text())
    if development.get("scope") != "development" or development["source_changed_during_run"]:
        raise ValueError("Require completed frozen development results")
    training = json.loads((args.checkpoint / "report.json").read_text())
    if training.get("source_changed", training.get("source_changed_during_run", True)):
        raise ValueError("Require valid completed training")
    files = [Path(__file__), args.development, args.source,
             *args.checkpoint.glob("*.json"), *args.checkpoint.glob("*.safetensors"),
             *[Path(__file__).with_name(name) for name in (
                 "name_adapted_adapter.py", "name_encoder_adaptation.py",
                 "name_generalization_model.py", "name_coverage_decoding.py",
                 "name_bioes_ablation.py", "evaluate_name_generalization.py",
                 "train_name_span_experiment.py")],
             *Path("src/ko_pii_guard").glob("*.py")]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    cases = read_selected_cases(args.source, development["selected_ids"])
    if any(normalize_text(case["text"]).text != case["text"] for case in cases):
        raise ValueError("Identity normalization required for reusable coordinate cache")
    torch.set_num_threads(2)
    ner = build_ner(dict(checkpoint=str(args.checkpoint), device=args.device,
                        threshold=0., decoder="viterbi"))
    logits, raw_rows = [], []
    ner.on_logits = lambda value: logits.append(value.detach().cpu().clone())

    class Recorder:
        def analyze(self, text):
            results = ner.analyze(text)
            raw_rows.append([dict(entity=r.entity_type, start=r.start, end=r.end, score=r.score)
                             for r in results])
            return results

    guard = KoreanPIIGuard(entities=["KR_NAME"], score_threshold=0., ner=Recorder())
    predictions = []
    for index, case in enumerate(cases, 1):
        predictions.append({(r.start, r.end) for r in guard.analyze(case["text"])
                            if r.entity == "KR_NAME"})
        if index % 100 == 0 or index == len(cases):
            print(f"cached adapted development {index}/{len(cases)}", flush=True)
    if len(logits) != len(cases) or len(raw_rows) != len(cases):
        raise ValueError("Missing original-coordinate cached outputs")
    torch.save(dict(selected_ids=development["selected_ids"], logits=logits,
                    frozen_sha256=hashes), args.cache)
    args.spans.write_text("".join(json.dumps(dict(id=c["id"], spans=r)) + "\n"
                                  for c, r in zip(cases, raw_rows, strict=True)))
    report = dict(scope="development_adapted_encoder_diagnostic", frozen_sha256=hashes,
                  cache=str(args.cache), spans=str(args.spans),
                  metrics=metrics(cases, predictions),
                  source_changed_during_run=any(hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                              != h for p, h in hashes.items()))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print({k: v for k, v in report["metrics"].items() if k != "errors"}, flush=True)
    if report["source_changed_during_run"]:
        raise ValueError("Diagnostic inputs changed during inference")


if __name__ == "__main__":
    main()
