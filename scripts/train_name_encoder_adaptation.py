"""Fixed three-epoch real-data LoRA adaptation with original frozen E5 priors.

Reads only prepared train/validation and original cached priors, never external
evaluation data. Checkpoints contain adapter tensors and the small name head.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from name_encoder_adaptation import inject_lora, load_lora_state_dict, lora_state_dict
from name_generalization_model import GeneralNameHead, decode_char_logits, encode_character_features
from safetensors.torch import load_file, save_file
from torch.nn.utils.rnn import pad_sequence
from train_name_generalization import MODEL_ID, REVISION, digest, gold_labels, load_rows

THRESHOLDS = (0.5, 0.7, 0.9)


def tokenize_rows(tokenizer, cases, max_length):
    tokens = tokenizer(
        [case["text"] for case in cases],
        return_offsets_mapping=True,
        truncation=False,
        padding=False,
    )
    rows = []
    for i in range(len(cases)):
        if len(tokens["input_ids"][i]) > max_length:
            raise ValueError("Source sentence exceeds encoder; refusing truncation")
        rows.append({key: value[i] for key, value in tokens.items()})
    return rows


def forward_batch(model, head, tokenizer, tokens, cases, cached, *, device):
    offsets = [row["offset_mapping"] for row in tokens]
    inputs = tokenizer.pad(
        [{k: v for k, v in row.items() if k != "offset_mapping"} for row in tokens],
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    # Encoder LoRA receives gradients; original prior is never recomputed here.
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
    ):
        hidden = model.roberta(**inputs).last_hidden_state
    features, positions = [], []
    for i, case in enumerate(cases):
        f, p = encode_character_features(case["text"], hidden[i, : len(offsets[i])], offsets[i])
        features.append(f.half())
        positions.append(p.float())
    lengths = torch.tensor([len(case["text"]) for case in cases])
    chars = pad_sequence([row[1] for row in cached], batch_first=True).to(device)
    prior = pad_sequence([row[3] for row in cached], batch_first=True).to(device)
    target = pad_sequence([row[4] for row in cached], batch_first=True, padding_value=-100).to(
        device
    )
    logits = head(
        pad_sequence(features, batch_first=True),
        chars,
        pad_sequence(positions, batch_first=True),
        prior,
        lengths,
    )
    return logits, lengths, target


def evaluate(model, head, tokenizer, tokens, cases, cached, *, device):
    head.eval()
    totals = {str(threshold): [0, 0, 0, 0] for threshold in THRESHOLDS}
    with torch.inference_mode():
        for start in range(0, len(cases), 32):
            batch = cases[start : start + 32]
            logits, lengths, _ = forward_batch(
                model,
                head,
                tokenizer,
                tokens[start : start + 32],
                batch,
                cached[start : start + 32],
                device=device,
            )
            logits = logits.float().cpu()
            for i, case in enumerate(batch):
                expected = {
                    (e["start"], e["end"]) for e in case["expected"] if e["entity"] == "KR_NAME"
                }
                spans = decode_char_logits(logits[i, : lengths[i]], threshold=0.0)
                for threshold in THRESHOLDS:
                    predictions = {(s, e) for s, e, score in spans if score >= threshold}
                    covered = {j for s, e in predictions for j in range(s, e)}
                    counts = (
                        len(expected & predictions),
                        len(predictions - expected),
                        len(expected - predictions),
                        sum(set(range(s, e)) <= covered for s, e in expected),
                    )
                    for j, count in enumerate(counts):
                        totals[str(threshold)][j] += count
    return {
        key: dict(
            tp=v[0],
            fp=v[1],
            fn=v[2],
            fully_covered_names=v[3],
            f1=2 * v[0] / (2 * v[0] + v[1] + v[2]) if v[0] else 0.0,
        )
        for key, v in totals.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--initial", type=Path, default=Path("artifacts/name-generalization-v1"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="One batch; never writes a candidate")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new/empty output; refuse to overwrite an experiment")
    torch.set_num_threads(2)
    random.seed(20261012)
    torch.manual_seed(20261012)
    train_path, val_path = args.data_dir / "train.jsonl", args.data_dir / "validation.jsonl"
    train, val = load_rows(train_path), load_rows(val_path)
    config = json.loads((args.initial / "config.json").read_text())
    if config["model_id"] != MODEL_ID or config["revision"] != REVISION:
        raise ValueError("Initial checkpoint has a different encoder")
    if json.loads((args.initial / "report.json").read_text()).get("source_changed") is not False:
        raise ValueError("Initial model did not finish with frozen inputs")
    vocabulary = config["vocabulary"]
    print("Loading and validating original frozen-prior cache", flush=True)
    payload = torch.load(args.cache, weights_only=True, mmap=True)
    for path in (train_path, val_path):
        if payload["key"]["files"].get(str(path.resolve())) != digest(path):
            raise ValueError("Cached priors differ from prepared source data")
    if (
        payload["key"]["encoder"] != [MODEL_ID, REVISION]
        or payload["key"]["vocabulary"] != vocabulary
    ):
        raise ValueError("Cached encoder/vocabulary differs")
    cached = payload["rows"][: len(train) + len(val)]
    if len(cached) != len(train) + len(val):
        raise ValueError("Incomplete original-prior cache")
    for case, row in zip([*train, *val], cached, strict=True):
        if not torch.equal(row[4], gold_labels(case)) or len(row[3]) != len(case["text"]):
            raise ValueError("Original-prior cache labels/offsets disagree")
    sources = [
        Path(__file__),
        Path(__file__).with_name("name_encoder_adaptation.py"),
        Path(__file__).with_name("name_generalization_model.py"),
        Path(__file__).with_name("name_bioes_ablation.py"),
        Path(__file__).with_name("train_name_generalization.py"),
        train_path,
        val_path,
        args.data_dir / "manifest.json",
        args.initial / "head.safetensors",
        args.initial / "config.json",
        args.initial / "report.json",
        args.cache,
    ]
    print("Hashing frozen sources and 2.2 GB feature cache", flush=True)
    hashes = {str(path.resolve()): digest(path) for path in sources}
    print("Loading pinned E5 and initializing LoRA", flush=True)
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    options = dict(revision=REVISION, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, **options)
    model = (
        AutoModelForTokenClassification.from_pretrained(MODEL_ID, use_safetensors=True, **options)
        .to(args.device)
        .eval()
    )
    layers = inject_lora(model)
    head = GeneralNameHead(
        char_vocab_size=len(vocabulary) + 2,
        **{
            key: config[key]
            for key in ("hidden_size", "projection_size", "char_embedding_size", "lstm_hidden_size")
        },
    ).to(args.device)
    head.load_state_dict(load_file(str(args.initial / "head.safetensors")))
    trainable = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        [dict(params=trainable, lr=1e-4), dict(params=head.parameters(), lr=5e-4)]
    )
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 4.0, 4.0, 4.0, 4.0], device=args.device)
    )
    selected_cases = train[:32] if args.smoke else [*train, *val]
    tokens = tokenize_rows(
        tokenizer,
        selected_cases,
        min(tokenizer.model_max_length, model.config.max_position_embeddings),
    )
    if args.smoke:
        head.train()
        logits, _, targets = forward_batch(
            model, head, tokenizer, tokens, selected_cases, cached[:32], device=args.device
        )
        loss = loss_fn(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        if not all(value.grad is not None for value in trainable):
            raise ValueError("Missing LoRA gradient through character expansion")
        if not any(value.grad.abs().sum() > 0 for value in trainable):
            raise ValueError("Zero LoRA gradients")
        if any(value.grad is not None for value in model.parameters() if not value.requires_grad):
            raise ValueError("Frozen encoder received gradients")
        optimizer.step()
        print(
            json.dumps(
                dict(
                    smoke_passed=True,
                    loss=float(loss.detach()),
                    lora_parameters=sum(value.numel() for value in trainable),
                )
            )
        )
        return
    args.output.mkdir(parents=True, exist_ok=True)
    adaptation = dict(rank=8, alpha=16, last_layers=4, modules=layers)
    manifest = dict(
        source_sha256=hashes,
        model_id=MODEL_ID,
        revision=REVISION,
        seed=20261012,
        epochs=3,
        thresholds=THRESHOLDS,
        batch_size=32,
        lora_learning_rate=1e-4,
        head_learning_rate=5e-4,
        name_class_weight=4.0,
        encoder_dropout=False,
        encoder_autocast="bfloat16",
        original_prior="Immutable original E5 probabilities from frozen feature cache",
        adaptation=adaptation,
        train_rows=len(train),
        validation_rows=len(val),
        selection="real validation exact F1, fewer FP, higher threshold, earlier epoch",
        runtime_promotion=False,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    history, best = [], None
    started = time.monotonic()
    for epoch in range(1, 4):
        head.train()
        indices = list(range(len(train)))
        random.shuffle(indices)
        loss_sum = 0.0
        for start in range(0, len(indices), 32):
            ids = indices[start : start + 32]
            optimizer.zero_grad(set_to_none=True)
            logits, _, targets = forward_batch(
                model,
                head,
                tokenizer,
                [tokens[i] for i in ids],
                [train[i] for i in ids],
                [cached[i] for i in ids],
                device=args.device,
            )
            loss = loss_fn(logits.flatten(0, 1), targets.flatten())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite adaptation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*trainable, *head.parameters()], 1.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            if start % (32 * 100) == 0:
                print(
                    json.dumps(
                        dict(
                            epoch=epoch,
                            trained=min(start + 32, len(train)),
                            loss=float(loss.detach()),
                            elapsed=time.monotonic() - started,
                        )
                    ),
                    flush=True,
                )
        scores = evaluate(
            model,
            head,
            tokenizer,
            tokens[len(train) :],
            val,
            cached[len(train) :],
            device=args.device,
        )
        record = dict(epoch=epoch, loss_sum=loss_sum, validation=scores)
        history.append(record)
        print(json.dumps(record), flush=True)
        for threshold, score in scores.items():
            rank = (score["f1"], -score["fp"], float(threshold), -epoch)
            if best is None or rank > best[0]:
                best = (rank, epoch, float(threshold), score)
                save_file(lora_state_dict(model), str(args.output / "lora.safetensors"))
                save_file(
                    {
                        name: value.detach().cpu().contiguous()
                        for name, value in head.state_dict().items()
                    },
                    str(args.output / "head.safetensors"),
                )
    load_lora_state_dict(model, load_file(str(args.output / "lora.safetensors")))
    config.update(
        adaptation=adaptation,
        threshold=best[2],
        selected_epoch=best[1],
        encoder_autocast="bfloat16",
        original_prior_required=True,
    )
    (args.output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )
    report = dict(
        history=history,
        selected_epoch=best[1],
        threshold=best[2],
        validation=best[3],
        elapsed_seconds=time.monotonic() - started,
        source_changed=any(digest(path) != expected for path, expected in hashes.items()),
        runtime_promotion=False,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}), flush=True)
    if report["source_changed"]:
        raise RuntimeError("Frozen adaptation inputs changed")


if __name__ == "__main__":
    main()
