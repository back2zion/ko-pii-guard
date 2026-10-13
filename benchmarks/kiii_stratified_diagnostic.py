"""Summarize pinned Kiii² synthetic errors by variation and document length.

This is a diagnostic on an already inspected test split, not a fresh holdout.
Only aggregate counts and source digests are written; no text or document IDs.
The pinned dataset must already exist in the local Hugging Face cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

from ko_pii_guard import KoreanPIIGuard
from ko_pii_guard.analyzer import DEFAULT_ENTITIES

if __package__:
    from .private_corpus_benchmark import (
        covered_by_predictions,
        is_ignored_prediction,
        source_hash,
    )
else:
    from private_corpus_benchmark import (
        covered_by_predictions,
        is_ignored_prediction,
        source_hash,
    )

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_kiii_evaluation import (  # noqa: E402
    DATA_SHA256,
    DATASET_ID,
    REVISION,
    convert,
    digest_file,
    selected_rows,
)


def summarize(rows: list[dict], guard: KoreanPIIGuard) -> dict:
    groups: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in rows:
        case = convert(row, include_name_address=False)
        value = case["text"]
        gold = {(item["type"], item["start"], item["end"]) for item in case["entities"]}
        ignored = {(item["start"], item["end"]) for item in case["ignore"]}
        all_predicted = {(item.entity, item.start, item.end) for item in guard.analyze(value)}
        predicted = {
            item for item in all_predicted
            if not is_ignored_prediction(item, gold, ignored)
        }
        masked = guard.mask(value, style="stars")
        fully_masked = sum(
            covered_by_predictions(start, end, predicted)
            and masked[start:end] == "*" * (end - start)
            for _, start, end in gold
        )
        for group in (("variation", row["variation_level"]),
                      ("length", row["context_len_bucket"])):
            counts = groups[group]
            counts["documents"] += 1
            counts["gold"] += len(gold)
            counts["tp"] += len(gold & predicted)
            counts["fp"] += len(predicted - gold)
            counts["fn"] += len(gold - predicted)
            counts["fully_masked"] += fully_masked
    return {
        f"{kind}:{value}": {
            **counts,
            "exact_f1": (2 * counts["tp"] / (2 * counts["tp"] + counts["fp"] + counts["fn"])
                         if 2 * counts["tp"] + counts["fp"] + counts["fn"] else None),
            "full_mask_coverage": (counts["fully_masked"] / counts["gold"]
                                   if counts["gold"] else None),
        }
        for (kind, value), counts in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-frozen", action="store_true",
                        help="require the documented baseline aggregate")
    args = parser.parse_args()
    source = Path(hf_hub_download(DATASET_ID, "data/test.jsonl", repo_type="dataset",
                                  revision=REVISION, local_files_only=True))
    if digest_file(source) != DATA_SHA256:
        raise ValueError("Pinned Kiii² source hash changed")
    rows = selected_rows(source)
    groups = summarize(rows, KoreanPIIGuard())
    totals = {key: sum(group[key] for name, group in groups.items()
                       if name.startswith("variation:"))
              for key in ("documents", "gold", "tp", "fp", "fn", "fully_masked")}
    if args.check_frozen and totals != {
        "documents": 60, "gold": 1456, "tp": 938, "fp": 111,
        "fn": 518, "fully_masked": 951,
    }:
        raise RuntimeError("Stratified totals differ from the frozen aggregate")
    scorer = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("private_corpus_benchmark.py"),
                 ROOT / "scripts" / "prepare_kiii_evaluation.py"):
        scorer.update(path.name.encode())
        scorer.update(path.read_bytes())
    report = {
        "scope": "synthetic_finance_diagnostic_after_test_inspection",
        "dataset_revision": REVISION,
        "dataset_sha256": DATA_SHA256,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "scorer_sha256": scorer.hexdigest(),
        "current_source_sha256": source_hash(ROOT),
        "entities": list(DEFAULT_ENTITIES),
        "totals": totals,
        "groups": groups,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
