"""Train a conservative candidate rejection stage while preserving the v3 decoder bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from name_address_benchmark import evaluate  # noqa: E402
from name_context_v5_benchmark import RecordedGuard, nonregression_gate  # noqa: E402
from train_name_context import collate, encode, validate_splits  # noqa: E402

from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard  # noqa: E402
from ko_pii_guard.name_context import (  # noqa: E402
    HEAD_LABELS,
    ContextSpanFilter,
    span_context_features,
)
from ko_pii_guard.ner import KoreanNER, _decode_tokens  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def samples(cases, cached, head, local_context=False):
    inputs, labels, required = [], [], []
    with torch.inference_mode():
        for base in range(0, len(cases), 64):
            f, c, p, lengths, _ = collate(cached[base:base+64])
            scores, classes = head(f, c, p, lengths).softmax(-1).max(-1)
            for row, case in enumerate(cases[base:base+64]):
                tokens = {(i, i+1): (0, HEAD_LABELS[int(classes[row, i])], float(scores[row, i]))
                          for i in range(len(case["text"]))}
                predicted = {(r.start, r.end) for r in _decode_tokens(tokens, .9)}
                gold = {(e["start"], e["end"]) for e in case["expected"]
                        if e["entity"] == "KR_NAME"}
                negative = predicted - gold
                # Every sampled span in a fully annotated non-person sentence
                # is negative. Sampling is training-only, with no runtime list.
                if not gold:
                    words = list(re.finditer(r"[가-힣A-Za-z]+", case["text"]))
                    for word in words[::max(1, len(words)//3)][:3]:
                        negative.add((word.start(), min(word.end(), word.start()+3)))
                spans = sorted(gold | negative)
                if not spans:
                    continue
                values = span_context_features(cached[base+row][0], spans, local_context)
                for span, value in zip(spans, values, strict=True):
                    inputs.append(value.clone())
                    labels.append(float(span not in gold))
                    required.append(not gold and span in predicted)
    return (torch.stack(inputs).clone(), torch.tensor(labels), torch.tensor(required))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "benchmarks/data/name_context_v5.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--local-context", action="store_true")
    parser.add_argument("--intermediate-size", type=int, default=64)
    parser.add_argument("--cosine-decay", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new or empty")
    torch.set_num_threads(2)
    torch.manual_seed(20261010)
    random.seed(20261010)
    torch.use_deterministic_algorithms(True)
    cases = [json.loads(line) for line in args.data.read_text().splitlines()]
    splits = validate_splits(cases)
    base = ROOT / "artifacts/name-context-v3"
    sources = [Path(__file__), ROOT / "scripts/train_name_context.py",
               ROOT / "benchmarks/name_context_v5_cases.py",
               ROOT / "benchmarks/name_context_v6_cases.py",
               ROOT / "benchmarks/name_context_v5_benchmark.py",
               ROOT / "benchmarks/name_address_benchmark.py",
               *sorted((ROOT / "src/ko_pii_guard").glob("*.py"))]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    base_hashes = {name: digest(base / name) for name in (
        "name_context.safetensors", "name_context_config.json"
    )}
    data_hash = digest(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(data_sha256=data_hash, source_sha256=hashes, baseline_sha256=base_hashes,
                    seed=20261010, epochs=args.epochs, threshold=.99, learning_rate=.001,
                    split_counts={s: len(rows) for s, rows in splits.items()},
                    local_context=args.local_context, intermediate_size=args.intermediate_size,
                    cosine_decay=args.cosine_decay,
                    selection="zero rejected train/validation gold, reject every known non-person "
                              "baseline false span; maximum validation rejections; earliest tie"
                    )
    (args.output / "training_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    ner = KoreanNER.from_pretrained(name_context_path=base)
    for module in (ner.model, ner.name_context_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    train_cache = encode(splits["train"], ner, 64)
    val_cache = encode(splits["validation"], ner, 64)
    x, y, required = samples(
        splits["train"], train_cache, ner.name_context_head, args.local_context
    )
    vx, vy, _ = samples(splits["validation"], val_cache, ner.name_context_head, args.local_context)
    del train_cache, val_cache
    model = ContextSpanFilter(intermediate_size=args.intermediate_size,
                              local_context=args.local_context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=.00005
    ) if args.cosine_decay else None
    weights = torch.where(y == 0, 3*len(y)/(2*(y == 0).sum()), len(y)/(2*(y == 1).sum()))
    best, best_state, best_epoch, history = None, None, None, []
    for epoch in range(1, args.epochs+1):
        model.train()
        order = torch.randperm(len(y))
        for offset in range(0, len(order), 256):
            indices = order[offset:offset+256]
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                model(x[indices]), y[indices], reduction="none"
            ) * weights[indices]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            train_probs = torch.cat([model(x[b:b+512]).sigmoid()
                                     for b in range(0, len(x), 512)])
            val_probs = model(vx).sigmoid()
        metrics = dict(epoch=epoch,
                       train_person_rejected=int(((train_probs >= .99) & (y == 0)).sum()),
                       validation_person_rejected=int(((val_probs >= .99) & (vy == 0)).sum()),
                       known_negative_left=int(((train_probs < .99) & required).sum()),
                       validation_negative_rejected=int(((val_probs >= .99) & (vy == 1)).sum()))
        eligible = (metrics["train_person_rejected"] == 0
                    and metrics["validation_person_rejected"] == 0
                    and metrics["known_negative_left"] == 0)
        metrics["eligible"] = eligible
        history.append(metrics)
        key = metrics["validation_negative_rejected"]
        if eligible and (best is None or key > best):
            best, best_epoch = key, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        print(json.dumps(metrics), flush=True)
        if scheduler is not None:
            scheduler.step()
    report = dict(manifest=manifest, selected_epoch=best_epoch, history=history)
    if best_state is None:
        report.update(rejected=True, evaluation_run=False)
        (args.output / "rejected_training_report.json").write_text(
            json.dumps(report, indent=2)+"\n"
        )
        raise RuntimeError("No safe filter; fresh evaluation was not run")
    model.load_state_dict(best_state)
    model.eval()
    shutil.copyfile(base / "name_context.safetensors", args.output / "name_context.safetensors")
    config = json.loads((base / "name_context_config.json").read_text())
    config["span_filter"] = dict(hidden_size=768, intermediate_size=args.intermediate_size,
                                 threshold=.99, local_context=args.local_context)
    (args.output / "name_context_config.json").write_text(json.dumps(config, indent=2)+"\n")
    save_file(best_state, str(args.output / "span_filter.safetensors"))
    guard = RecordedGuard(KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=ner))
    report["evaluation_baseline"] = evaluate(splits["evaluation"], guard)
    previous = guard.predictions
    ner.name_context_head.span_filter = model
    guard = RecordedGuard(KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=ner))
    report["evaluation_candidate"] = evaluate(splits["evaluation"], guard)
    report["span_gate"] = nonregression_gate(splits["evaluation"], previous, guard.predictions)
    before, after = (report[key]["counts"] for key in (
        "evaluation_baseline", "evaluation_candidate"
    ))
    report["accepted"] = report["span_gate"]["nonregression_passed"] and (
        after["false_positive_sentences"] < before["false_positive_sentences"]
        if before["false_positive_sentences"] else after["false_positive_sentences"] == 0
    )
    report["source_changed_during_run"] = (
        digest(args.data) != data_hash
        or any(digest(ROOT / name) != expected for name, expected in hashes.items())
        or any(digest(base / name) != expected for name, expected in base_hashes.items())
    )
    (args.output / "training_report.json").write_text(json.dumps(report, ensure_ascii=False,
                                                               indent=2)+"\n")
    if report["source_changed_during_run"] or not report["accepted"]:
        raise RuntimeError("Filter rejected by strict fresh per-span nonregression gate")
    print("selected", best_epoch, "accepted", report["accepted"], flush=True)


if __name__ == "__main__":
    main()
