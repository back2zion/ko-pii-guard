"""Frozen synthetic name-context diagnosis; not a population accuracy estimate.

Compare the pinned baseline with a benchmark-only lexicon score ablation.
The lexicon contains calibration positive surfaces only. No model retraining,
candidate creation, boundary edits or production policy changes are performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from presidio_analyzer import RecognizerResult

from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard
from ko_pii_guard.normalization import normalize_text

# Support both direct CLI execution and import by evaluation tests.
if __package__:
    from .name_address_benchmark import _dependency_versions, _source_metadata, load_cases
else:
    from name_address_benchmark import _dependency_versions, _source_metadata, load_cases

DATA = Path(__file__).parent / "data/name_context.jsonl"
THRESHOLD = 0.9
BOOST = 0.05


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration_lexicon(cases: list[dict]) -> frozenset[str]:
    return frozenset(case["surface"] for case in cases if case["split"] == "calibration"
                     and any(span["entity"] == "KR_NAME" for span in case["expected"]))


class RawCache:
    """One raw model call per normalized input shared by every comparison."""

    def __init__(self, model):
        self.model = model
        self.rows: dict[str, tuple] = {}

    def get(self, text: str) -> tuple:
        if text not in self.rows:
            self.rows[text] = tuple((r.entity_type, r.start, r.end, r.score)
                                    for r in self.model.analyze(text))
        return self.rows[text]


class ScoreOnlyNER:
    """Filter immutable raw candidates after an optional exact-surface boost."""

    def __init__(self, cache: RawCache, lexicon: frozenset[str], boost: float):
        self.cache, self.lexicon, self.boost = cache, lexicon, boost

    def scored(self, text: str) -> list[tuple]:
        return [(entity, start, end, min(1.0, score + self.boost)
                 if entity == "KR_NAME" and text[start:end] in self.lexicon else score)
                for entity, start, end, score in self.cache.get(text)]

    def analyze(self, text: str) -> list[RecognizerResult]:
        return [RecognizerResult(entity, start, end, score)
                for entity, start, end, score in self.scored(text) if score >= THRESHOLD]


def observe(case: dict, guard: KoreanPIIGuard) -> dict:
    text = case["text"]
    gold = {(r["entity"], r["start"], r["end"]) for r in case["expected"]}
    predicted = {(r.entity, r.start, r.end) for r in guard.analyze(text)}
    masked = guard.mask(text, style="stars")
    if len(masked) != len(text):
        raise ValueError("stars masking must preserve original length")
    changed = {i for i, (before, after) in enumerate(zip(text, masked, strict=True))
               if before != after}
    gold_chars = {i for _, start, end in gold for i in range(start, end)}
    covered = {span for span in gold if masked[span[1]:span[2]] == "*" * (span[2] - span[1])}
    partial, overwide, missing, boundary_error, split = 0, 0, 0, 0, 0
    for entity, start, end in gold:
        overlaps = [(s, e) for label, s, e in predicted
                    if label == entity and s < end and e > start]
        union = {i for s, e in overlaps for i in range(max(start, s), min(end, e))}
        partial += bool(union) and len(union) < end - start
        overwide += any(s < start or e > end for s, e in overlaps)
        missing += not overlaps
        boundary_error += bool(overlaps) and (entity, start, end) not in predicted
        split += len(overlaps) > 1 and (entity, start, end) not in predicted
    return {
        "predicted": sorted(predicted),
        "correct": gold == predicted,
        "counts": {
            "sentences": 1, "gold_spans": len(gold),
            "true_positive": len(gold & predicted), "false_positive": len(predicted - gold),
            "false_negative": len(gold - predicted), "fully_masked_gold": len(covered),
            "partial_gold": partial, "overwide_gold": overwide, "missed_gold": missing,
            "boundary_error_gold": boundary_error, "split_gold": split,
            "exact_sentences": int(gold == predicted),
            "negative_sentences": int(not gold),
            "false_positive_sentences": int(not gold and bool(predicted)),
            "negative_characters": len(text) if not gold else 0,
            "masked_negative_characters": len(changed) if not gold else 0,
            "masked_non_gold_characters": len(changed - gold_chars),
        },
    }


def metrics(rows: list[dict]) -> dict:
    counts: Counter = Counter()
    for row in rows:
        counts.update(row["counts"])
    tp, fp, fn = (counts[key] for key in ("true_positive", "false_positive", "false_negative"))

    def ratio(n, d):
        return n / d if d else None

    return {
        "counts": dict(counts),
        "exact_span_precision": ratio(tp, tp + fp),
        "exact_span_recall": ratio(tp, tp + fn),
        "exact_span_f1": ratio(2 * tp, 2 * tp + fp + fn),
        "full_mask_coverage": ratio(counts["fully_masked_gold"], counts["gold_spans"]),
        "negative_masked_character_ratio": ratio(counts["masked_negative_characters"],
                                                 counts["negative_characters"]),
    }


def paired_metrics(cases: list[dict], observations: dict[str, dict]) -> dict:
    groups: dict[str, list] = defaultdict(list)
    frames: dict[str, list] = defaultdict(list)
    for case in cases:
        groups[case["group_id"]].append(case)
        if case["track"] == "person_context":
            frames[case["context_id"]].append(case)
    contrasts = [rows for rows in groups.values()
                 if any(c["expected"] for c in rows) and any(not c["expected"] for c in rows)]

    def all_correct(rows):
        return all(observations[c["id"]]["correct"] for c in rows)

    return {
        "person_nonperson_surface_groups": len(contrasts),
        "all_correct_surface_groups": sum(all_correct(rows) for rows in contrasts),
        "person_context_frames": len(frames),
        "all_correct_frames_across_surfaces": sum(all_correct(rows) for rows in frames.values()),
        "definition": "all sentences in each group must have the exact gold span set",
    }


def summarize(cases: list[dict], observations: dict[str, dict]) -> dict:
    return {
        **metrics(list(observations.values())),
        "per_split": {
            split: {
                **metrics([observations[c["id"]] for c in cases if c["split"] == split]),
                "pairs": paired_metrics([c for c in cases if c["split"] == split], observations),
                **{
                    "by_" + field: {
                        value: metrics([observations[c["id"]] for c in cases
                                        if c["split"] == split and c[field] == value])
                        for value in sorted({c[field] for c in cases if c["split"] == split})
                    } for field in ("track", "name_family")
                },
            } for split in sorted({c["split"] for c in cases})
        },
        "error_ids": [c["id"] for c in cases if not observations[c["id"]]["correct"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = load_cases(args.data)
    lexicon = calibration_lexicon(cases)
    if not lexicon:
        parser.error("Calibration positive surfaces are required")
    if lexicon & {c["surface"] for c in cases if c["split"] == "challenge"}:
        parser.error("Unseen-surface challenge overlaps the calibration lexicon")
    source = _source_metadata()
    frozen = {"data": sha256(args.data), "benchmark": sha256(Path(__file__)),
              "generator": sha256(Path(__file__).with_name("name_context_cases.py")),
              "shared_evaluator": sha256(Path(__file__).with_name("name_address_benchmark.py"))}
    frozen_at = datetime.now(timezone.utc).isoformat()
    import torch

    from ko_pii_guard.ner import MODELS, KoreanNER

    torch.set_num_threads(2)
    model = KoreanNER.from_pretrained(score_threshold=0.0, device="cpu")
    cache = RawCache(model)
    backends = {"baseline": ScoreOnlyNER(cache, frozenset(), 0.0),
                "lexicon_score_only": ScoreOnlyNER(cache, lexicon, BOOST)}
    guards = {key: KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=backend)
              for key, backend in backends.items()}
    observations: dict[str, dict] = {key: {} for key in guards}
    raw_rows, promotions = [], []
    started = time.perf_counter()
    for i, case in enumerate(cases):
        for key, guard in guards.items():
            observations[key][case["id"]] = observe(case, guard)
        normalized = normalize_text(case["text"]).text
        raw = cache.get(normalized)
        boosted = backends["lexicon_score_only"].scored(normalized)
        if [r[:3] for r in raw] != [r[:3] for r in boosted]:
            raise AssertionError("Score-only ablation changed candidates or boundaries")
        hits = [r for r in raw if r[0] == "KR_NAME" and normalized[r[1]:r[2]] in lexicon]
        promoted = [dict(entity=r[0], start=r[1], end=r[2], raw_score=r[3], score=b[3],
                         surface=normalized[r[1]:r[2]])
                    for r, b in zip(raw, boosted, strict=True) if r[3] < THRESHOLD <= b[3]]
        before, after = (observations[key][case["id"]] for key in guards)
        if promoted:
            promotions.append({"id": case["id"], "split": case["split"],
                               "candidates": promoted,
                               "baseline_correct": before["correct"],
                               "boosted_correct": after["correct"],
                               "tp_change": after["counts"]["true_positive"]
                               - before["counts"]["true_positive"],
                               "fp_change": after["counts"]["false_positive"]
                               - before["counts"]["false_positive"]})
        raw_rows.append({"id": case["id"], "split": case["split"],
                         "surface_in_lexicon": case["surface"] in lexicon,
                         "raw_normalized_candidates": raw, "lexicon_hits": len(hits),
                         "near_threshold_candidates": sum(
                             THRESHOLD - BOOST <= r[3] < THRESHOLD for r in raw),
                         "results": {key: rows[case["id"]] for key, rows in observations.items()}})
        if (i + 1) % 50 == 0:
            print(f"Evaluated {i + 1}/{len(cases)}", file=sys.stderr, flush=True)
    if source != _source_metadata() or frozen != {
        "data": sha256(args.data), "benchmark": sha256(Path(__file__)),
        "generator": sha256(Path(__file__).with_name("name_context_cases.py")),
        "shared_evaluator": sha256(Path(__file__).with_name("name_address_benchmark.py")),
    }:
        raise RuntimeError("Evaluation inputs changed during inference")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output.with_suffix(".predictions.jsonl")
    predictions_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                        for row in raw_rows), encoding="utf-8")
    report = {
        "scope": "synthetic developmental contrast diagnosis, not independent population rates",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": ["uv", "run", "--frozen", "python", *sys.argv],
        "python": platform.python_version(), "platform": platform.platform(),
        "dependency_versions": _dependency_versions(), **source,
        "frozen_sha256": frozen, "frozen_at_utc": frozen_at,
        "inputs_unchanged_during_inference": True,
        "predictions_file": predictions_path.name, "predictions_sha256": sha256(predictions_path),
        "model": {"id": MODELS["e5"][0], "revision": MODELS["e5"][1],
                  "raw_threshold": 0.0, "final_threshold": THRESHOLD, "device": model.device,
                  "max_length": model.max_length, "stride": model.stride,
                  "batch_size": model.batch_size, "torch_threads": torch.get_num_threads(),
                  "local_files_only": True},
        "guard_threshold": guards["baseline"].score_threshold,
        "entities": list(SUPPORTED_ENTITIES),
        "lexicon": {"source": "calibration positive case surfaces only",
                    "surfaces": sorted(lexicon),
                    "sha256": hashlib.sha256(json.dumps(sorted(lexicon), ensure_ascii=False)
                                             .encode()).hexdigest(),
                    "boost": BOOST, "score_cap": 1.0, "match": "exact normalized candidate slice"},
        "corpus": {
            split: {"sentences": len(rows), "positive_sentences": sum(bool(c["expected"])
                                                                        for c in rows),
                    "negative_sentences": sum(not c["expected"] for c in rows),
                    "unique_surfaces": len({c["surface"] for c in rows}),
                    "unique_contexts": len({c["context_id"] for c in rows}),
                    "surface_groups": len({c["group_id"] for c in rows})}
            for split in sorted({c["split"] for c in cases})
            for rows in [[c for c in cases if c["split"] == split]]
        },
        "results": {key: summarize(cases, rows) for key, rows in observations.items()},
        "ablation": {
            split: {"lexicon_hit_candidates": sum(r["lexicon_hits"] for r in rows),
                    "near_threshold_candidates": sum(r["near_threshold_candidates"] for r in rows),
                    "promoted_candidates": sum(len(p["candidates"]) for p in promotions
                                               if p["split"] == split),
                    "corrected_sentences": sum(not r["results"]["baseline"]["correct"]
                                               and r["results"]["lexicon_score_only"]["correct"]
                                               for r in rows),
                    "harmed_sentences": sum(r["results"]["baseline"]["correct"]
                                            and not r["results"]["lexicon_score_only"]["correct"]
                                            for r in rows)}
            for split in sorted({c["split"] for c in cases})
            for rows in [[r for r in raw_rows if r["split"] == split]]
        },
        "promotions": promotions, "unique_model_inputs": len(cache.rows),
        "seconds_inference_and_cached_comparisons": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output), "ablation": report["ablation"],
                      "baseline": {s: r["counts"] for s, r in
                                   report["results"]["baseline"]["per_split"].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
