"""Refine rescue and non-person heads together; fresh evaluation follows all dev gates."""

from __future__ import annotations

import argparse
import copy
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
from name_context_v6_benchmark import RecordedGuard, nonregression_gate  # noqa: E402
from train_name_context import collate, encode, score_head, validate_splits  # noqa: E402

from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard  # noqa: E402
from ko_pii_guard.name_context import (  # noqa: E402
    HEAD_LABELS,
    ContextSpanFilter,
    load_name_head,
    span_context_features,
)
from ko_pii_guard.ner import MODEL_ID, MODEL_REVISION, KoreanNER, _decode_tokens  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_samples(cases, cached, primary, rescue):
    """Train from gold and both decoders' wrong spans, including mixed-person prose."""
    inputs, labels, required = [], [], []
    with torch.inference_mode():
        for base in range(0, len(cases), 64):
            f, c, p, lengths, _ = collate(cached[base:base+64])
            outputs = [head(f, c, p, lengths).softmax(-1).max(-1)
                       for head in (primary, rescue)]
            for row, case in enumerate(cases[base:base+64]):
                predicted = set()
                for (scores, classes), threshold in zip(outputs, (.9, .99), strict=True):
                    tokens = {(i, i+1): (0, HEAD_LABELS[int(classes[row, i])],
                                         float(scores[row, i])) for i in range(len(case["text"]))}
                    predicted.update((r.start, r.end) for r in _decode_tokens(tokens, threshold))
                gold = {(e["start"], e["end"]) for e in case["expected"]
                        if e["entity"] == "KR_NAME"}
                negative = predicted - gold
                if not gold:
                    words = list(re.finditer(r"[가-힣A-Za-z]+", case["text"]))
                    for word in words[::max(1, len(words)//3)][:3]:
                        negative.add((word.start(), min(word.end(), word.start()+3)))
                spans = sorted(gold | negative)
                if spans:
                    values = span_context_features(cached[base+row][0], spans, local_context=True)
                    for span, value in zip(spans, values, strict=True):
                        inputs.append(value.clone())
                        labels.append(float(span not in gold))
                        required.append(span in predicted - gold)
    return torch.stack(inputs).clone(), torch.tensor(labels), torch.tensor(required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "benchmarks/data/name_context_v8.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rescue-epochs", type=int, default=30)
    parser.add_argument("--filter-epochs", type=int, default=240)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be new or empty")
    if min(args.rescue_epochs, args.filter_epochs) < 1:
        parser.error("epochs must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(20261010)
    random.seed(20261010)
    torch.use_deterministic_algorithms(True)
    base = ROOT / "artifacts/name-context-v7"
    cases = [json.loads(line) for line in args.data.read_text().splitlines()]
    splits = validate_splits(cases)
    regressions = [json.loads(line) for version in (6, 7) for line in (
        ROOT / f"benchmarks/data/name_context_v{version}_regressions.jsonl"
    ).read_text().splitlines()]
    train_texts = {c["text"] for c in splits["train"]}
    if not all(c["text"] in train_texts for c in regressions):
        raise ValueError("Every development regression must be retired into training")
    sources = [Path(__file__), ROOT / "scripts/train_name_context.py",
               ROOT / "benchmarks/name_address_benchmark.py",
               ROOT / "benchmarks/name_context_v6_benchmark.py",
               ROOT / "benchmarks/name_context_v8_cases.py",
               *sorted((ROOT / "src/ko_pii_guard").glob("*.py"))]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    data_hashes = {str(p.relative_to(ROOT)): digest(p) for p in [args.data, *[
        ROOT / f"benchmarks/data/name_context_v{v}_regressions.jsonl" for v in (6, 7)
    ]]}
    base_hashes = {p.name: digest(p) for p in base.iterdir()
                   if p.suffix == ".safetensors" or p.name == "name_context_config.json"}
    manifest = dict(seed=20261010, split_counts={s: len(v) for s, v in splits.items()},
                    source_sha256=hashes, data_sha256=data_hashes, baseline_sha256=base_hashes,
                    rescue_epochs=args.rescue_epochs, filter_epochs=args.filter_epochs,
                    rescue_learning_rate=.0002, filter_learning_rate=.0002,
                    rescue_threshold=.99, filter_threshold=.99,
                    rescue_selection="zero FP/FN on all v6/v7 development cases, validation F1, "
                                     "fewer validation FP; earliest tie",
                    filter_selection="zero rejected train/validation gold, reject every predicted "
                                     "train false span including mixed prose; most validation "
                                     "negative rejections; earliest tie",
                    fresh_gate="after public dev zero: preserve every baseline-correct span; "
                               "no newly introduced false span")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    report = dict(manifest=manifest, rescue_history=[], filter_history=[], evaluation_run=False)

    def unchanged():
        return (all(digest(ROOT / name) == value for name, value in hashes.items())
                and all(digest(ROOT / name) == value for name, value in data_hashes.items())
                and all(digest(base / name) == value for name, value in base_hashes.items()))

    def reject(reason):
        report.update(accepted=False, rejected_reason=reason,
                      source_changed_during_run=not unchanged())
        (args.output / "rejected_training_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)+"\n"
        )
        raise RuntimeError(reason)

    ner = KoreanNER.from_pretrained(name_context_path=base)
    for parameter in ner.model.parameters():
        parameter.requires_grad_(False)
    train = encode(splits["train"], ner, 64)
    validation = encode(splits["validation"], ner, 64)
    by_text = {case["text"]: row for case, row in zip(splits["train"], train, strict=True)}
    regression_cache = [by_text[c["text"]] for c in regressions]
    rescue = copy.deepcopy(ner.name_context_head.rescue_head)
    optimizer = torch.optim.AdamW(rescue.parameters(), lr=.0002, weight_decay=.01)
    weights = torch.tensor([.25, 1., 1., 1., 1.])
    best, best_state = None, None
    rng = random.Random(20261010)
    for epoch in range(1, args.rescue_epochs+1):
        order = list(range(len(train)))
        rng.shuffle(order)
        rescue.train()
        for offset in range(0, len(order), 32):
            f, c, p, lengths, labels = collate([train[i] for i in order[offset:offset+32]])
            loss = torch.nn.functional.cross_entropy(
                rescue(f, c, p, lengths).reshape(-1, 5), labels.reshape(-1),
                weight=weights, ignore_index=-100,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rescue.parameters(), 1.)
            optimizer.step()
        val = score_head(rescue, validation, splits["validation"], .99)
        dev = score_head(rescue, regression_cache, regressions, .99)
        eligible = dev["fp"] == dev["fn"] == 0
        report["rescue_history"].append(dict(epoch=epoch, validation=val, development=dev,
                                             eligible=eligible))
        key = (val["f1"], -val["fp"])
        if eligible and (best is None or key > best):
            best, best_state = key, copy.deepcopy(rescue.state_dict())
            report["selected_rescue_epoch"] = epoch
        print(json.dumps(dict(stage="rescue", **report["rescue_history"][-1])), flush=True)
    if best_state is None:
        reject("No rescue checkpoint passed all known development cases; fresh eval not run")
    rescue.load_state_dict(best_state)
    rescue.eval()
    x, y, required = candidate_samples(splits["train"], train, ner.name_context_head, rescue)
    vx, vy, _ = candidate_samples(splits["validation"], validation, ner.name_context_head, rescue)
    del train, validation, by_text, regression_cache
    model = ContextSpanFilter(local_context=True)
    model.load_state_dict(ner.name_context_head.span_filter.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0002)
    weights = torch.where(y == 0, 3*len(y)/(2*(y == 0).sum()), len(y)/(2*(y == 1).sum()))
    best, best_filter = None, None
    for epoch in range(1, args.filter_epochs+1):
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
            probs = torch.cat([model(x[b:b+512]).sigmoid() for b in range(0, len(x), 512)])
            val_probs = model(vx).sigmoid()
        metrics = dict(epoch=epoch,
                       train_person_rejected=int(((probs >= .99) & (y == 0)).sum()),
                       validation_person_rejected=int(((val_probs >= .99) & (vy == 0)).sum()),
                       known_false_spans_left=int(((probs < .99) & required).sum()),
                       validation_negative_rejected=int(((val_probs >= .99) & (vy == 1)).sum()))
        eligible = (metrics["train_person_rejected"] == 0
                    and metrics["validation_person_rejected"] == 0
                    and metrics["known_false_spans_left"] == 0)
        report["filter_history"].append(dict(**metrics, eligible=eligible))
        key = metrics["validation_negative_rejected"]
        if eligible and (best is None or key > best):
            best, best_filter = key, copy.deepcopy(model.state_dict())
            report["selected_filter_epoch"] = epoch
        print(json.dumps(dict(stage="filter", **report["filter_history"][-1])), flush=True)
    if best_filter is None:
        reject("No filter preserved all gold and removed every known false span; "
               "fresh eval not run")
    model.load_state_dict(best_filter)
    model.eval()
    shutil.copyfile(base / "name_context.safetensors", args.output / "name_context.safetensors")
    shutil.copyfile(base / "name_context_config.json", args.output / "name_context_config.json")
    save_file(best_state, str(args.output / "rescue_name_context.safetensors"))
    save_file(best_filter, str(args.output / "span_filter.safetensors"))
    ner.name_context_head = load_name_head(
        args.output, model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu"
    )
    report["development_candidate"] = evaluate(
        regressions, KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=ner)
    )
    dev = report["development_candidate"]["counts"]
    if dev["false_positive"] or dev["false_negative"] or dev["fully_covered_spans"] != 468:
        reject("Public development masking gate failed; fresh evaluation not run")
    if not unchanged():
        reject("Frozen inputs changed; fresh evaluation not run")
    guards = {}
    for profile, path in (("baseline", base), ("candidate", args.output)):
        ner.name_context_head = load_name_head(
            path, model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu"
        )
        guard = RecordedGuard(KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=ner))
        report[f"evaluation_{profile}"] = evaluate(splits["evaluation"], guard)
        guards[profile] = guard.predictions
    report["evaluation_run"] = True
    report["span_gate"] = nonregression_gate(
        splits["evaluation"], guards["baseline"], guards["candidate"]
    )
    report["accepted"] = report["span_gate"]["nonregression_passed"]
    report["source_changed_during_run"] = not unchanged()
    (args.output / "training_report.json").write_text(json.dumps(report, ensure_ascii=False,
                                                               indent=2)+"\n")
    if not report["accepted"] or not unchanged():
        raise RuntimeError("Fresh per-span nonregression gate rejected candidate")
    print("Accepted", report["selected_rescue_epoch"], report["selected_filter_epoch"], flush=True)


if __name__ == "__main__":
    main()
