"""Train a general character decoder from real source-separated training material.

Only the provided train and validation files are read. Frozen E5 representations
and probabilities remain local. The external evaluation is a separate command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from name_generalization_model import (
    GeneralNameHead,
    character_prior,
    decode_char_logits,
    encode_character_features,
)
from safetensors.torch import save_file
from torch.nn.utils.rnn import pad_sequence

MODEL_ID = "FrameByFrame/korean-pii-e5-base"
REVISION = "a308c54b4407819624a5661e31e162a269f39818"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def gold_labels(case):
    target = torch.zeros(len(case["text"]), dtype=torch.long)
    for span in case["expected"]:
        if span["entity"] != "KR_NAME":
            continue
        start, end = span["start"], span["end"]
        if not 0 <= start < end <= len(target) or target[start:end].any():
            raise ValueError("Invalid or overlapping gold name")
        if end - start == 1:
            target[start] = 4
        else:
            target[start], target[end - 1] = 1, 3
            target[start + 1:end - 1] = 2
    return target


def encode(rows, vocabulary, *, device, batch_size=24):
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    options = dict(revision=REVISION, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, **options)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID, use_safetensors=True, **options
    ).to(device).eval()
    cached = []
    with torch.inference_mode():
        for base in range(0, len(rows), batch_size):
            batch = rows[base:base + batch_size]
            encoded = tokenizer([r["text"] for r in batch], return_offsets_mapping=True,
                                truncation=False, padding=True, return_tensors="pt")
            offsets = encoded.pop("offset_mapping").tolist()
            if encoded["input_ids"].shape[1] > min(
                tokenizer.model_max_length, model.config.max_position_embeddings
            ):
                raise ValueError("Training input exceeds encoder; refusing truncation")
            outputs = model(**{k: v.to(device) for k, v in encoded.items()},
                            output_hidden_states=True)
            probs = outputs.logits.softmax(-1).cpu()
            hidden = outputs.hidden_states[-1].cpu()
            for row, case in enumerate(batch):
                f, p = encode_character_features(case["text"], hidden[row], offsets[row])
                prior = character_prior(probs[row], offsets[row], 0, len(case["text"]),
                                        model.config.id2label)
                cached.append((f.half().cpu(), torch.tensor([
                    vocabulary.get(c, 1) for c in case["text"]
                ]), p.cpu(), prior.cpu(), gold_labels(case)))
            if base % (batch_size * 20) == 0:
                print(f"encoded {min(base + batch_size, len(rows))}/{len(rows)}", flush=True)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return [tuple(t.clone() for t in row) for row in cached]


def collate(rows, device):
    features, chars, positions, priors, targets = zip(*rows, strict=True)
    lengths = torch.tensor([len(c) for c in chars])
    values = [pad_sequence(parts, batch_first=True).to(device)
              for parts in (features, chars, positions, priors)]
    target = pad_sequence(targets, batch_first=True, padding_value=-100).to(device)
    return (*values, lengths, target)


def evaluate(head, cached, cases, *, device, thresholds):
    totals = {str(t): [0, 0, 0, 0] for t in thresholds}
    head.eval()
    with torch.inference_mode():
        for base in range(0, len(cached), 64):
            f, c, p, prior, lengths, _ = collate(cached[base:base + 64], device)
            logits = head(f, c, p, prior, lengths).cpu()
            for i, case in enumerate(cases[base:base + 64]):
                expected = {(s["start"], s["end"]) for s in case["expected"]
                            if s["entity"] == "KR_NAME"}
                spans = decode_char_logits(logits[i, :lengths[i]], threshold=0.)
                for t in thresholds:
                    predicted = {(s, e) for s, e, score in spans if score >= t}
                    covered = {j for s, e in predicted for j in range(s, e)}
                    values = totals[str(t)]
                    for index, value in enumerate((len(expected & predicted),
                                                   len(predicted - expected),
                                                   len(expected - predicted),
                                                   sum(set(range(s, e)) <= covered
                                                       for s, e in expected))):
                        values[index] += value
    return {t: dict(tp=v[0], fp=v[1], fn=v[2], fully_covered_names=v[3],
                    f1=2 * v[0] / (2 * v[0] + v[1] + v[2]) if v[0] else 0.)
            for t, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    if args.epochs < 1 or (args.output.exists() and any(args.output.iterdir())):
        parser.error("Positive epochs and new/empty output required")
    torch.set_num_threads(2)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    files = [args.data_dir / "train.jsonl", args.data_dir / "validation.jsonl"]
    synthetic_validation = args.data_dir / "synthetic-validation.jsonl"
    if synthetic_validation.exists():
        files.append(synthetic_validation)
    files.extend([Path(__file__), Path(__file__).with_name("name_generalization_model.py"),
                  Path(__file__).with_name("name_bioes_ablation.py")])
    hashes = {str(p.resolve()): digest(p) for p in files}
    train, val = load_rows(files[0]), load_rows(files[1])
    synthetic = load_rows(synthetic_validation) if synthetic_validation.exists() else []
    if {r["text"] for r in train} & {r["text"] for r in [*val, *synthetic]}:
        raise ValueError("Train/validation text overlap")
    vocabulary = {c: i + 2 for i, c in enumerate(sorted({c for r in train for c in r["text"]}))}
    thresholds = [.5, .7, .9, .95]
    config = dict(model_id=MODEL_ID, revision=REVISION, vocabulary=vocabulary,
                  hidden_size=768, projection_size=64, char_embedding_size=32,
                  lstm_hidden_size=64)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(source_sha256=hashes, seed=args.seed, epochs=args.epochs,
                    model=config, train_rows=len(train), validation_rows=len(val),
                    synthetic_validation_rows=len(synthetic), thresholds=thresholds,
                    selection="real validation exact F1, fewer FP, higher threshold, earlier epoch",
                    batch_size=64, learning_rate=.001, name_class_weight=2.,
                    device=args.device, runtime_promotion=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    cache_key = {"files": {str(p.resolve()): digest(p) for p in files[:len(files) - 3]},
                 "encoder": [MODEL_ID, REVISION], "vocabulary": vocabulary,
                 "encoding_source": digest(Path(__file__)),
                 "character_encoding_source": digest(
                     Path(__file__).with_name("name_generalization_model.py"))}
    if args.cache.exists():
        payload = torch.load(args.cache, weights_only=True)
        if payload["key"] != cache_key:
            raise ValueError("Feature cache differs from frozen inputs")
        cached = payload["rows"]
    else:
        cached = encode([*train, *val, *synthetic], vocabulary, device=args.device)
        torch.save(dict(key=cache_key, rows=cached), args.cache)
    tc, vc, sc = cached[:len(train)], cached[len(train):len(train) + len(val)], cached[
        len(train) + len(val):]
    head = GeneralNameHead(char_vocab_size=len(vocabulary) + 2).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 2., 2., 2., 2.],
                                                           device=args.device))
    history, best = [], None
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = list(range(len(tc)))
        random.shuffle(order)
        total_loss = 0.
        for base in range(0, len(order), 64):
            f, c, p, prior, lengths, target = collate([tc[i] for i in order[base:base + 64]],
                                                      args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(f, c, p, prior, lengths)
            loss = loss_fn(logits.flatten(0, 1), target.flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
            total_loss += float(loss.detach())
        scores = evaluate(head, vc, val, device=args.device, thresholds=thresholds)
        record = dict(epoch=epoch, loss=total_loss, validation=scores)
        history.append(record)
        print(json.dumps(record), flush=True)
        for threshold, score in scores.items():
            rank = (score["f1"], -score["fp"], float(threshold), -epoch)
            if best is None or rank > best[0]:
                best = (rank, epoch, float(threshold), score)
                save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()},
                          str(args.output / "head.safetensors"))
    from safetensors.torch import load_file
    head.load_state_dict(load_file(str(args.output / "head.safetensors")))
    synthetic_scores = evaluate(head, sc, synthetic, device=args.device,
                                 thresholds=[best[2]]) if synthetic else {}
    config.update(threshold=best[2], selected_epoch=best[1])
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    report = dict(history=history, selected_epoch=best[1], threshold=best[2], validation=best[3],
                  synthetic_validation=synthetic_scores, elapsed_seconds=time.monotonic() - started,
                  source_changed=any(digest(p) != h for p, h in hashes.items()),
                  runtime_promotion=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}), flush=True)
    if report["source_changed"]:
        raise RuntimeError("Frozen training inputs changed")


if __name__ == "__main__":
    main()
