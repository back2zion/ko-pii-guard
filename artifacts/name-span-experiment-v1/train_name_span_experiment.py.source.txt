"""Run a fixed-budget, three-seed research experiment; never promote to runtime.

Validation chooses checkpoints/thresholds. Previously inspected evaluation rows
are diagnostic development data, not independent generalization evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from name_span_experiment import SpanNameHead, decoded_spans, span_loss
from safetensors.torch import save_file
from torch.nn.utils.rnn import pad_sequence
from train_name_context import encode, validate_splits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from name_address_benchmark import load_cases  # noqa: E402

from ko_pii_guard import KoreanPIIGuard  # noqa: E402
from ko_pii_guard.ner import KoreanNER  # noqa: E402

THRESHOLDS = (0.3, 0.5, 0.7, 0.9)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gold(case):
    return {(e["start"], e["end"]) for e in case["expected"] if e["entity"] == "KR_NAME"}


def batch(rows, cases, vocabulary, width):
    lengths = torch.tensor([len(c["text"]) for c in cases])
    features = pad_sequence([r[0] for r in rows], batch_first=True)
    chars = pad_sequence(
        [torch.tensor([vocabulary.get(c, 1) for c in case["text"]]) for case in cases],
        batch_first=True,
    )
    positions = pad_sequence([r[2] for r in rows], batch_first=True)
    targets = torch.zeros(len(rows), features.shape[1], width)
    for i, case in enumerate(cases):
        for start, end in gold(case):
            if end - start > width:
                raise ValueError("Gold name exceeds research candidate width; cannot truncate")
            targets[i, start, end - start - 1] = 1
    return features, chars, positions, lengths, targets


def metrics(cases, predictions):
    tp = fp = fn = covered = extra = 0
    errors = []
    for case, predicted in zip(cases, predictions, strict=True):
        expected = gold(case)
        tp += len(expected & predicted)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
        actual_chars = {i for start, end in predicted for i in range(start, end)}
        gold_chars = {i for start, end in expected for i in range(start, end)}
        covered += sum(set(range(start, end)) <= actual_chars for start, end in expected)
        extra += len(actual_chars - gold_chars)
        if expected != predicted:
            errors.append(
                dict(
                    id=case["id"],
                    missed=sorted(expected - predicted),
                    false=sorted(predicted - expected),
                )
            )
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        f1=2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        fully_covered_names=covered,
        unnecessary_masked_characters=extra,
        errors=errors,
    )


def predict(head, cached, cases, vocabulary, width, thresholds=THRESHOLDS):
    result = {t: [] for t in thresholds}
    head.eval()
    with torch.inference_mode():
        for base in range(0, len(cases), 32):
            values = batch(cached[base : base + 32], cases[base : base + 32], vocabulary, width)
            logits, valid = head(*values[:4], max_span_width=width)
            for t in thresholds:
                result[t].extend(
                    {(s.start, s.end) for s in decoded_spans(row_logits, row_valid, threshold=t)}
                    for row_logits, row_valid in zip(logits, valid, strict=True)
                )
    return result


def gate(cases, baseline, predictions):
    lost, introduced, exposed = [], [], []
    for case, old, new in zip(cases, baseline, predictions, strict=True):
        expected = gold(case)
        lost.extend((case["id"], *span) for span in sorted((expected & old) - new))
        introduced.extend(
            (case["id"], *span) for span in sorted((new - expected) - (old - expected))
        )
        old_chars = {i for s, e in old for i in range(s, e)}
        new_chars = {i for s, e in new for i in range(s, e)}
        exposed.extend(
            (case["id"], s, e)
            for s, e in sorted(expected)
            if set(range(s, e)) <= old_chars and not set(range(s, e)) <= new_chars
        )
    return dict(
        lost_correct_spans=lost,
        introduced_false_spans=introduced,
        newly_exposed_names=exposed,
        passed=not (lost or introduced or exposed),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "benchmarks/data/name_context_v11.jsonl"
    )
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/name-context-v11")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, default=Path("/tmp/name-span-features-v1.pt"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261010, 20261011, 20261012])
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive epochs and unique seeds required")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new or empty output; do not overwrite experiments")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cases = load_cases(args.data)
    splits = validate_splits(cases)
    # Deduplicate exact training documents; reject contradictory annotations.
    unique = {}
    for case in splits["train"]:
        if case["text"] in unique and gold(unique[case["text"]]) != gold(case):
            raise ValueError("Conflicting gold for the same training document")
        unique.setdefault(case["text"], case)
    train, validation = list(unique.values()), splits["validation"]
    diagnostic = splits["evaluation"]
    vocabulary = {c: i + 2 for i, c in enumerate(sorted({c for r in train for c in r["text"]}))}
    width = 32
    sources = [
        Path(__file__),
        ROOT / "scripts/name_span_experiment.py",
        ROOT / "scripts/train_name_context.py",
        ROOT / "benchmarks/name_address_benchmark.py",
        *sorted((ROOT / "src/ko_pii_guard").glob("*.py")),
    ]
    files = [
        args.data,
        *sources,
        *args.baseline.glob("*.safetensors"),
        args.baseline / "name_context_config.json",
    ]
    hashes = {str(p): digest(p) for p in files}
    manifest = dict(
        source_and_input_sha256=hashes,
        seeds=args.seeds,
        epochs=args.epochs,
        train_rows=len(train),
        validation_rows=len(validation),
        diagnostic_rows=len(diagnostic),
        max_span_width=width,
        thresholds=THRESHOLDS,
        learning_rate=0.001,
        positive_weight=16.0,
        batch_size=64,
        frozen_encoder=True,
        initialization="v11 primary projection/LSTM/character weights; new span scorer",
        selection="validation exact F1, fewer FP, higher threshold, earlier epoch",
        diagnostic_is_previously_inspected=True,
        automatic_runtime_promotion=False,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    ner = KoreanNER.from_pretrained(name_context_path=args.baseline)
    cache_key = dict(
        data_sha256=digest(args.data),
        encoder_source_sha256={
            str(p): digest(p)
            for p in (
                ROOT / "src/ko_pii_guard/ner.py",
                ROOT / "src/ko_pii_guard/name_context.py",
                ROOT / "scripts/train_name_context.py",
            )
        },
        model_revision=ner_module_revision(),
        train_ids=[c["id"] for c in train],
        validation_ids=[c["id"] for c in validation],
    )
    if args.feature_cache.exists():
        cache = torch.load(args.feature_cache, weights_only=True)
        if cache["key"] != cache_key:
            raise ValueError("Feature cache metadata mismatch")
        training, valid_cache = cache["train"], cache["validation"]
        del cache
    else:
        print("Encoding training and validation; no diagnostic rows yet", flush=True)
        training, valid_cache = encode(train, ner, 64), encode(validation, ner, 64)
        torch.save(dict(key=cache_key, train=training, validation=valid_cache), args.feature_cache)
    report = dict(
        manifest=manifest,
        feature_cache_sha256=digest(args.feature_cache),
        seeds=[],
        accepted_for_runtime=False,
        limitation="Synthetic development only; no independent external assessment",
    )
    selected = []
    for seed in args.seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        head = SpanNameHead(len(vocabulary) + 2)
        primary = ner.name_context_head
        head.projection.load_state_dict(primary.projection.state_dict())
        head.sequence.load_state_dict(primary.sequence.state_dict())
        with torch.no_grad():
            for char, index in vocabulary.items():
                head.characters.weight[index].copy_(primary.characters.weight[ord(char) % 4096])
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
        history, best_key, best_state, best_epoch, best_threshold = [], None, None, None, None
        for epoch in range(1, args.epochs + 1):
            head.train()
            order = list(range(len(train)))
            random.shuffle(order)
            losses = []
            for base in range(0, len(order), 64):
                indices = order[base : base + 64]
                values = batch(
                    [training[i] for i in indices], [train[i] for i in indices], vocabulary, width
                )
                optimizer.zero_grad()
                logits, valid = head(*values[:4], max_span_width=width)
                loss = span_loss(logits, valid, values[4], positive_weight=16.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            predictions = predict(head, valid_cache, validation, vocabulary, width)
            scores = {t: metrics(validation, p) for t, p in predictions.items()}
            for threshold, score in scores.items():
                key = (score["f1"], -score["fp"], threshold, -epoch)
                if best_key is None or key > best_key:
                    best_key, best_state = key, copy.deepcopy(head.state_dict())
                    best_epoch, best_threshold = epoch, threshold
            summary = dict(
                epoch=epoch,
                loss=sum(losses) / len(losses),
                validation={
                    str(t): {k: v for k, v in s.items() if k != "errors"} for t, s in scores.items()
                },
            )
            history.append(summary)
            print(json.dumps(dict(seed=seed, **summary)), flush=True)
        head.load_state_dict(best_state)
        seed_path = args.output / f"seed-{seed}"
        seed_path.mkdir()
        save_file(
            {k: v.contiguous() for k, v in best_state.items()}, str(seed_path / "head.safetensors")
        )
        config = dict(
            vocabulary=vocabulary,
            max_span_width=width,
            threshold=best_threshold,
            selected_epoch=best_epoch,
            architecture=dict(vocabulary_size=len(vocabulary) + 2),
        )
        (seed_path / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n"
        )
        selected.append((head, best_threshold))
        report["seeds"].append(
            dict(seed=seed, selected_epoch=best_epoch, threshold=best_threshold, history=history)
        )
        (args.output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
    print("All checkpoints frozen; start previously inspected diagnostic", flush=True)
    diagnostic_cache = encode(diagnostic, ner, 64)
    baseline_guard = KoreanPIIGuard(entities=["KR_NAME"], ner=ner)
    baseline = [
        {(r.start, r.end) for r in baseline_guard.analyze(c["text"]) if r.entity == "KR_NAME"}
        for c in diagnostic
    ]
    report["baseline_diagnostic"] = metrics(diagnostic, baseline)
    for record, (head, threshold) in zip(report["seeds"], selected, strict=True):
        prediction = predict(
            head, diagnostic_cache, diagnostic, vocabulary, width, thresholds=[threshold]
        )[threshold]
        record["diagnostic"] = metrics(diagnostic, prediction)
        record["diagnostic_gate"] = gate(diagnostic, baseline, prediction)
        print(
            "DIAGNOSTIC",
            record["seed"],
            {k: v for k, v in record["diagnostic"].items() if k != "errors"},
            "gate",
            record["diagnostic_gate"]["passed"],
            flush=True,
        )
    report["source_changed_during_run"] = any(digest(Path(p)) != h for p, h in hashes.items())
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    if report["source_changed_during_run"]:
        raise SystemExit("Frozen inputs changed during experiment")


def ner_module_revision():
    from ko_pii_guard.ner import MODEL_REVISION

    return MODEL_REVISION


if __name__ == "__main__":
    main()
