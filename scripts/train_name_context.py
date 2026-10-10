"""Train an opt-in character name head locally, with a frozen E5 backbone.

Only train/validation rows are encoded before checkpoint selection. The held-out
split is evaluated once after selection. This is synthetic engineering validation,
not a population accuracy estimate. No raw training text is sent over a network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from name_address_benchmark import evaluate  # noqa: E402
from name_context_v6_benchmark import RecordedGuard, nonregression_gate  # noqa: E402

from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard  # noqa: E402
from ko_pii_guard.name_context import (  # noqa: E402
    CHAR_BUCKETS,
    FORMAT_VERSION,
    HEAD_LABELS,
    ContextNameHead,
    character_features,
    load_name_head,
)
from ko_pii_guard.ner import MODEL_ID, MODEL_REVISION, KoreanNER, _decode_tokens  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gold_labels(case):
    labels = torch.zeros(len(case["text"]), dtype=torch.long)
    for span in case["expected"]:
        if span["entity"] != "KR_NAME":
            continue
        start, end = span["start"], span["end"]
        if not 0 <= start < end <= len(labels):
            raise ValueError("Invalid gold offsets")
        if labels[start:end].any():
            raise ValueError("Overlapping gold names")
        if end - start == 1:
            labels[start] = 4
        else:
            labels[start], labels[end - 1] = 1, 3
            labels[start + 1 : end - 1] = 2
    return labels


def encode(cases, ner, batch_size=16):
    features = []
    with torch.inference_mode():
        for base in range(0, len(cases), batch_size):
            batch = cases[base : base + batch_size]
            encoded = ner.tokenizer(
                [c["text"] for c in batch],
                return_offsets_mapping=True,
                padding=True,
                return_tensors="pt",
                truncation=False,
            )
            offsets = encoded.pop("offset_mapping").tolist()
            if encoded["input_ids"].shape[1] > ner.max_length:
                raise ValueError("Training sentence exceeds model window; do not silently truncate")
            hidden = ner.model.base_model(**encoded).last_hidden_state
            for row, case in enumerate(batch):
                values = character_features(case["text"], hidden[row], offsets[row])
                features.append((*[v.clone() for v in values], gold_labels(case)))
            if base % 256 == 0:
                print(f"encoded {min(base + batch_size, len(cases))}/{len(cases)}", flush=True)
    # Tensors created in inference mode cannot be saved for backward. Clone after
    # leaving that context; keep cached E5 representations in float16 on CPU.
    return [
        (f.to(torch.float16).clone(), c.clone(), p.clone(), y.clone()) for f, c, p, y in features
    ]


def collate(rows):
    lengths = torch.tensor([len(row[1]) for row in rows])
    features, chars, positions, labels = zip(*rows, strict=True)
    return (
        pad_sequence(features, batch_first=True),
        pad_sequence(chars, batch_first=True),
        pad_sequence(positions, batch_first=True),
        lengths,
        pad_sequence(labels, batch_first=True, padding_value=-100),
    )


def score_head(head, cached, cases, threshold, *, collect_spans=False, protected=None):
    tp = fp = fn = 0
    matched, negative_errors = set(), set()
    head.eval()
    with torch.inference_mode():
        for base in range(0, len(cached), 32):
            f, c, p, lengths, _ = collate(cached[base : base + 32])
            scores, labels = head(f, c, p, lengths).softmax(-1).max(-1)
            for row, case in enumerate(cases[base : base + 32]):
                tokens = {
                    (i, i + 1): (0, HEAD_LABELS[int(labels[row, i])], float(scores[row, i]))
                    for i in range(len(case["text"]))
                }
                actual = {(r.start, r.end) for r in _decode_tokens(tokens, threshold)}
                gold = {
                    (e["start"], e["end"]) for e in case["expected"] if e["entity"] == "KR_NAME"
                }
                tp += len(actual & gold)
                fp += len(actual - gold)
                fn += len(gold - actual)
                matched.update((case["id"], start, end) for start, end in actual & gold)
                if not gold and actual:
                    negative_errors.add(case["id"])
    result = dict(tp=tp, fp=fp, fn=fn, f1=2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    if collect_spans or protected is not None:
        result["negative_false_positive_sentences"] = len(negative_errors)
    if collect_spans:
        result["matched_gold"] = sorted(matched)
    if protected is not None:
        result["protected_gold_lost"] = len(protected - matched)
    return result


def checkpoint_is_eligible(metrics, regression, baseline, require_zero):
    """A higher aggregate F1 cannot compensate for a lost protected gold span."""
    return (
        not require_zero
        or (regression is not None and regression["fp"] == 0 and regression["fn"] == 0)
    ) and (
        baseline is None
        or (metrics["protected_gold_lost"] == 0 and metrics["fp"] <= baseline["fp"]
            and metrics["negative_false_positive_sentences"]
            <= baseline["negative_false_positive_sentences"])
    )


def validate_splits(cases):
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    splits = {
        s: [c for c in cases if c["split"] == s] for s in ("train", "validation", "evaluation")
    }
    if sum(map(len, splits.values())) != len(cases) or not all(splits.values()):
        raise ValueError("Require nonempty train/validation/evaluation splits")
    for i, left in enumerate(splits):
        for right in list(splits)[i + 1 :]:
            for key in ("text", "context_id", "surface"):
                a = {c[key] for c in splits[left] if c[key] is not None}
                b = {c[key] for c in splits[right] if c[key] is not None}
                if a & b:
                    raise ValueError(f"Split overlap in {key}: {left}/{right}")

            # Multiple people and repeated names must all be considered, rather
            # than checking only a row's primary target name.
            def names(rows):
                return {
                    c["text"][e["start"] : e["end"]]
                    for c in rows
                    for e in c["expected"]
                    if e["entity"] == "KR_NAME"
                }

            if names(splits[left]) & names(splits[right]):
                raise ValueError(f"Split overlap in gold name surfaces: {left}/{right}")
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "benchmarks/data/name_context_training.jsonl"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--outside-weight", type=float, default=0.25)
    parser.add_argument("--encoding-batch-size", type=int, default=16)
    parser.add_argument("--nonregression-baseline", type=Path)
    parser.add_argument("--preserve-from", type=Path,
                        help="Preserve this decoder/filter and add the trained head as a rescue")
    parser.add_argument("--score-threshold", type=float, default=0.9)
    parser.add_argument("--require-regression-zero", action="store_true")
    parser.add_argument(
        "--regression-data",
        type=Path,
        action="append",
        default=[],
        help="Known development corpora used for checkpoint selection",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("epochs and threads must be positive")
    if not 0 < args.learning_rate <= 1 or not 0 < args.outside_weight <= 1:
        parser.error("learning rate and outside weight must be in (0, 1]")
    if args.encoding_batch_size < 1:
        parser.error("encoding batch size must be positive")
    if not 0 < args.score_threshold <= 1:
        parser.error("score threshold must be in (0, 1]")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be new or empty; preserve prior experiments")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    source_files = [
        Path(__file__),
        *sorted((ROOT / "src/ko_pii_guard").glob("*.py")),
        ROOT / "benchmarks/name_address_benchmark.py",
        ROOT / "benchmarks/name_context_training_cases.py",
        ROOT / "benchmarks/name_context_v2_cases.py",
        ROOT / "benchmarks/name_context_v3_cases.py",
        ROOT / "benchmarks/name_context_v4_cases.py",
        ROOT / "benchmarks/name_context_v5_cases.py",
        ROOT / "benchmarks/name_context_v6_cases.py",
        ROOT / "benchmarks/name_context_v7_cases.py",
        ROOT / "benchmarks/name_context_v6_benchmark.py",
    ]
    frozen = {str(p.relative_to(ROOT)): digest(p) for p in source_files}
    data_sha = digest(args.data)
    cases = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines()]
    splits = validate_splits(cases)
    regression_cases = [
        json.loads(line)
        for path in args.regression_data
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    # Regressions are development selection data, never part of the held-out
    # evaluation. Reject accidental access to evaluation text or names.
    if regression_cases:

        def targets(rows):
            return {
                c["text"][e["start"] : e["end"]]
                for c in rows
                for e in c["expected"]
                if e["entity"] == "KR_NAME"
            }

        if {c["text"] for c in regression_cases} & {
            c["text"] for c in splits["evaluation"]
        } or targets(regression_cases) & targets(splits["evaluation"]):
            raise ValueError("Regression selection data overlaps evaluation")
    regression_hashes = {str(path): digest(path) for path in args.regression_data}
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(
        format_version=FORMAT_VERSION,
        base_model_id=MODEL_ID,
        base_revision=MODEL_REVISION,
        labels=list(HEAD_LABELS),
        char_buckets=CHAR_BUCKETS,
        architecture=dict(hidden_size=768, projection_size=64, char_size=16, recurrent_size=64),
    )
    # This manifest is written BEFORE inference/training and records the frozen
    # corpus, split policy, architecture, and planned fixed run settings.
    manifest = dict(
        data_sha256=data_sha,
        source_sha256=frozen,
        seed=args.seed,
        epochs=args.epochs,
        threads=args.threads,
        batch_size=32,
        learning_rate=args.learning_rate,
        confidence_threshold=args.score_threshold,
        class_weights=[args.outside_weight, 1.0, 1.0, 1.0, 1.0],
        initialization_sha256={
            name: digest(args.initialize_from / name)
            for name in ("name_context_config.json", "name_context.safetensors")
        } if args.initialize_from else None,
        nonregression_baseline_sha256={
            name: digest(args.nonregression_baseline / name)
            for name in ("name_context_config.json", "name_context.safetensors")
        } if args.nonregression_baseline else None,
        require_regression_zero=args.require_regression_zero,
        preserved_checkpoint_sha256={
            p.name: digest(p) for p in args.preserve_from.iterdir()
            if p.suffix == ".safetensors" or p.name == "name_context_config.json"
        } if args.preserve_from else None,
        encoding_batch_size=args.encoding_batch_size,
        selected_by=(
            "fewest development regression errors, validation exact-span F1, "
            "then fewer validation FP; earliest tie"
            if regression_cases
            else "validation exact-span F1, then fewer FP; earliest tie"
        ),
        regression_data_sha256=regression_hashes,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        split_counts={s: len(rows) for s, rows in splits.items()},
        torch_version=torch.__version__,
        **config,
    )
    (args.output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    ner = KoreanNER.from_pretrained()
    for parameter in ner.model.parameters():
        parameter.requires_grad_(False)
    started = time.perf_counter()
    train = encode(splits["train"], ner, args.encoding_batch_size)
    validation = encode(splits["validation"], ner, args.encoding_batch_size)
    protected, validation_baseline = None, None
    if args.nonregression_baseline:
        reference = load_name_head(
            args.nonregression_baseline, model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu"
        )
        validation_baseline = score_head(
            reference, validation, splits["validation"], 0.9, collect_spans=True
        )
        protected = {tuple(span) for span in validation_baseline["matched_gold"]}
    # These are inspected development cases already present in training. Reuse
    # frozen representations while retaining each regression's own gold labels.
    by_text = {case["text"]: values for case, values in zip(splits["train"], train, strict=True)}
    if regression_cases and all(c["text"] in by_text for c in regression_cases):
        regression = [(*by_text[c["text"]][:3], gold_labels(c)) for c in regression_cases]
    else:
        regression = encode(regression_cases, ner) if regression_cases else []
    head = (
        load_name_head(
            args.initialize_from, model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu"
        )
        if args.initialize_from else ContextNameHead(**config["architecture"])
    )
    if args.initialize_from and config["architecture"] != json.loads(
        (args.initialize_from / "name_context_config.json").read_text()
    )["architecture"]:
        raise ValueError("Initialization architecture does not match training configuration")
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=0.01)
    weights = torch.tensor(manifest["class_weights"])
    best, best_epoch, history, best_weights = None, 0, [], None
    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        indices = list(range(len(train)))
        rng.shuffle(indices)
        head.train()
        loss_sum = 0.0
        for base in range(0, len(indices), 32):
            f, c, p, lengths, labels = collate([train[i] for i in indices[base : base + 32]])
            logits = head(f, c, p, lengths)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 5), labels.reshape(-1), weight=weights, ignore_index=-100
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach())
        metrics = score_head(head, validation, splits["validation"], args.score_threshold,
                             protected=protected)
        history.append(dict(epoch=epoch, loss_sum=loss_sum, **metrics))
        regression_metrics = (
            score_head(head, regression, regression_cases, args.score_threshold)
            if regression else None
        )
        key = (metrics["f1"], -metrics["fp"])
        if regression_metrics is not None:
            history[-1]["development_regression"] = regression_metrics
            key = (-(regression_metrics["fp"] + regression_metrics["fn"]), *key)
        eligible = checkpoint_is_eligible(
            metrics, regression_metrics, validation_baseline, args.require_regression_zero
        )
        if args.require_regression_zero or validation_baseline is not None:
            history[-1]["checkpoint_eligible"] = eligible
        if eligible and (best is None or key > best):
            best, best_epoch = key, epoch
            best_weights = {k: v.detach().clone() for k, v in head.state_dict().items()}
        print(json.dumps(history[-1]), flush=True)
    if best_weights is None:
        (args.output / "rejected_training_report.json").write_text(json.dumps(dict(
            manifest=manifest, validation_baseline=validation_baseline, history=history,
            rejected_reason="No checkpoint passed development and validation nonregression gates",
            evaluation_run=False,
        ), indent=2) + "\n")
        raise RuntimeError("No nonregressing checkpoint; held-out evaluation was not run")
    head.load_state_dict(best_weights)
    head.eval()
    save_file(best_weights, str(args.output / "name_context.safetensors"))
    (args.output / "name_context_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.preserve_from:
        shutil.move(args.output / "name_context.safetensors",
                    args.output / "rescue_name_context.safetensors")
        preserved_config = json.loads((args.preserve_from / "name_context_config.json").read_text())
        if "rescue_head" in preserved_config:
            raise ValueError("Nested rescue checkpoints are not supported")
        preserved_config["rescue_head"] = dict(architecture=config["architecture"],
                                                score_threshold=args.score_threshold)
        for name in manifest["preserved_checkpoint_sha256"]:
            if name.endswith(".safetensors"):
                shutil.copyfile(args.preserve_from / name, args.output / name)
        (args.output / "name_context_config.json").write_text(
            json.dumps(preserved_config, indent=2) + "\n"
        )
    report = dict(
        manifest=manifest,
        selected_epoch=best_epoch,
        history=history,
        validation_baseline=validation_baseline,
        training_seconds=time.perf_counter() - started,
        checkpoint_sha256=digest(args.output / "name_context.safetensors"),
    )
    # Public APIs and actual stars masking on held-out cases, not cached logits.
    if args.preserve_from:
        ner.name_context_head = load_name_head(
            args.preserve_from, model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu"
        )
    baseline = RecordedGuard(KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=ner))
    report["evaluation_baseline"] = evaluate(splits["evaluation"], baseline)
    ner.name_context_head = load_name_head(
        args.output, model_id=MODEL_ID, revision=MODEL_REVISION, device="cpu"
    ) if args.preserve_from else head
    refined = RecordedGuard(KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=ner))
    report["evaluation_candidate"] = evaluate(splits["evaluation"], refined)
    if args.preserve_from:
        report["span_gate"] = nonregression_gate(
            splits["evaluation"], baseline.predictions, refined.predictions
        )
        report["accepted"] = report["span_gate"]["nonregression_passed"]
    report["source_changed_during_run"] = (
        digest(args.data) != data_sha
        or any(digest(p) != frozen[str(p.relative_to(ROOT))] for p in source_files)
        or any(digest(p) != expected for p, expected in regression_hashes.items())
        or (args.initialize_from is not None and any(
            digest(args.initialize_from / name) != expected
            for name, expected in manifest["initialization_sha256"].items()
        ))
        or (args.nonregression_baseline is not None and any(
            digest(args.nonregression_baseline / name) != expected
            for name, expected in manifest["nonregression_baseline_sha256"].items()
        ))
        or (args.preserve_from is not None and any(
            digest(args.preserve_from / name) != expected
            for name, expected in manifest["preserved_checkpoint_sha256"].items()
        ))
    )
    if report["source_changed_during_run"]:
        raise RuntimeError("Frozen inputs changed during run; refusing final report")
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.preserve_from and not report["accepted"]:
        raise RuntimeError("Rescue checkpoint rejected by fresh per-span nonregression gate")
    print("selected epoch", best_epoch, "report", args.output / "training_report.json", flush=True)


if __name__ == "__main__":
    main()
