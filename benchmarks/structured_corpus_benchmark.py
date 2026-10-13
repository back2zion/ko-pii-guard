"""Compare whole-field JSON masking on a local annotated corpus without timing.

Inputs and model files stay local. Output contains aggregate counts and hashes,
never record IDs, paths inside records, or field values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from importlib.metadata import version
from pathlib import Path

if __package__:
    from .private_corpus_benchmark import (
        archive_revision,
        finite_float,
        freeze_input,
        reject_duplicate_keys,
        reject_nonfinite_number,
        source_hash,
        write_report,
    )
else:
    from private_corpus_benchmark import (
        archive_revision,
        finite_float,
        freeze_input,
        reject_duplicate_keys,
        reject_nonfinite_number,
        source_hash,
        write_report,
    )

ROOT = Path(__file__).resolve().parents[1]
HELPER = Path(__file__).with_name("private_corpus_benchmark.py")


def scorer_hash() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__).resolve(), HELPER):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_record(value: object) -> None:
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("record keys must be strings")
        for child in value.values():
            _validate_record(child)
    elif isinstance(value, list):
        for child in value:
            _validate_record(child)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("record contains a non-finite number")
    elif value is not None and type(value) not in (str, int, bool):
        raise ValueError("record contains unsupported data")


def _leaf_at(record: object, path: object) -> object:
    if not isinstance(path, list) or not path:
        raise ValueError("annotation path must be a nonempty list")
    node = record
    for part in path:
        if isinstance(node, dict) and isinstance(part, str) and part in node:
            node = node[part]
        elif (isinstance(node, list) and type(part) is int
              and 0 <= part < len(node)):
            node = node[part]
        else:
            raise ValueError("annotation path does not resolve")
    if type(node) not in (str, int) or not str(node):
        raise ValueError("annotation path must target a nonempty string or integer")
    return node


def read_cases(path: Path, allowed: set[str]) -> list[dict]:
    cases = []
    seen_ids = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line, object_pairs_hook=reject_duplicate_keys,
                                 parse_constant=reject_nonfinite_number,
                                 parse_float=finite_float)
                identifier = raw["id"]
                record = raw["record"]
                targets = raw["targets"]
                if (not isinstance(identifier, str) or not identifier
                        or identifier in seen_ids):
                    raise ValueError("id must be a unique nonempty string")
                if not isinstance(record, (dict, list)):
                    raise ValueError("record must be a JSON object or list")
                _validate_record(record)
                if not isinstance(targets, list):
                    raise ValueError("targets must be a list")
                gold = {}
                for target in targets:
                    if not isinstance(target, dict) or target.get("type") not in allowed:
                        raise ValueError("target type must be selected and supported")
                    path_key = tuple(target["path"])
                    value = _leaf_at(record, target["path"])
                    if (str(value).strip("*") == ""
                            or re.fullmatch(r"<[A-Z][A-Z0-9_]*>", str(value))):
                        raise ValueError("target field is already fully masked")
                    if path_key in gold:
                        raise ValueError("duplicate target path")
                    if "surface" in target and target["surface"] != str(value):
                        raise ValueError("target surface does not match the field")
                    gold[path_key] = target["type"]
                ignored = set()
                negatives = set()
                for key, destination in (("ignore_paths", ignored),
                                         ("negative_paths", negatives)):
                    items = raw.get(key, [])
                    if not isinstance(items, list):
                        raise ValueError(f"{key} must be a list")
                    for item in items:
                        path_key = tuple(item)
                        _leaf_at(record, item)
                        if path_key in destination:
                            raise ValueError("duplicate annotation path")
                        destination.add(path_key)
                if (set(gold) & ignored or set(gold) & negatives
                        or ignored & negatives):
                    raise ValueError("target, ignore and negative paths must be disjoint")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid input at line {line_number}: {error}") from None
            seen_ids.add(identifier)
            cases.append({"id": identifier, "record": record, "gold": gold, "ignore": ignored,
                          "negative": negatives})
    if not cases:
        raise ValueError("Input contains no records")
    return cases


def check_split_overlap(holdout: list[dict], development: list[dict]) -> None:
    """Reject shared IDs or identical records regardless of JSON key order."""
    if {case["id"] for case in holdout} & {case["id"] for case in development}:
        raise ValueError("Development and holdout inputs share a record ID")

    def fingerprints(cases: list[dict]) -> set[str]:
        return {
            hashlib.sha256(json.dumps(
                case["record"], ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            for case in cases
        }

    if fingerprints(holdout) & fingerprints(development):
        raise ValueError("Development and holdout inputs share a JSON record")


def paired_leaves(original: object, tagged: object, starred: object,
                  path: tuple = ()):
    """Walk output by original structure so masked keys remain observable."""
    if isinstance(original, dict):
        if not isinstance(tagged, dict) or not isinstance(starred, dict):
            raise RuntimeError("Masking changed object structure")
        if len(original) != len(tagged) or len(original) != len(starred):
            raise RuntimeError("Masking changed object field count")
        for (key, child), (tag_key, tag_child), (star_key, star_child) in zip(
            original.items(), tagged.items(), starred.items(), strict=True
        ):
            yield from paired_leaves(child, tag_child, star_child, path + (key,))
            if tag_key != key or star_key != key:
                yield path + (key,), None, None, None, True
    elif isinstance(original, list):
        if (not isinstance(tagged, list) or not isinstance(starred, list)
                or len(original) != len(tagged) or len(original) != len(starred)):
            raise RuntimeError("Masking changed list structure")
        for index, (child, tag_child, star_child) in enumerate(
            zip(original, tagged, starred, strict=True)
        ):
            yield from paired_leaves(child, tag_child, star_child, path + (index,))
    else:
        yield path, original, tagged, starred, False


def evaluate(cases: list[dict], guard) -> dict:
    counts = Counter({key: 0 for key in (
        "target_fields", "exact_type_fields", "fully_masked_fields",
        "nontarget_fields", "changed_nontarget_fields", "hard_negative_fields",
        "changed_hard_negative_fields", "ignored_fields", "changed_keys_out_of_scope",
        "positive_records", "positive_records_with_unmasked_target",
        "negative_records", "negative_records_with_changed_field",
    )})
    by_entity = {
        kind: Counter({"gold": 0, "exact_type": 0, "fully_masked": 0})
        for kind in guard.entities
    }
    wrong_type = Counter()
    for case in cases:
        record, gold = case["record"], case["gold"]
        tagged = guard.mask_json(record, style="tag")
        starred = guard.mask_json(record, style="stars")
        document_fp = False
        document_miss = False
        for path, original, tag_value, star_value, key_change in paired_leaves(
            record, tagged, starred
        ):
            if key_change:
                counts["changed_keys_out_of_scope"] += 1
                continue
            if path in case["ignore"]:
                counts["ignored_fields"] += 1
                continue
            changed = (tag_value != original or star_value != original)
            if path in gold:
                kind = gold[path]
                exact = tag_value == f"<{kind}>" and tag_value != original
                full = (star_value == "*" * len(str(original))
                        and star_value != original)
                if isinstance(tag_value, str):
                    other = re.fullmatch(r"<([A-Z_]+)>", tag_value)
                    if other and other.group(1) != kind:
                        wrong_type[(kind, other.group(1))] += 1
                counts["target_fields"] += 1
                counts["exact_type_fields"] += exact
                counts["fully_masked_fields"] += full
                by_entity[kind]["gold"] += 1
                by_entity[kind]["exact_type"] += exact
                by_entity[kind]["fully_masked"] += full
                document_miss |= not full
            else:
                counts["nontarget_fields"] += 1
                counts["changed_nontarget_fields"] += changed
                document_fp |= changed
                if path in case["negative"]:
                    counts["hard_negative_fields"] += 1
                    counts["changed_hard_negative_fields"] += changed
        if gold:
            counts["positive_records"] += 1
            counts["positive_records_with_unmasked_target"] += document_miss
        else:
            counts["negative_records"] += 1
            counts["negative_records_with_changed_field"] += document_fp
    return {
        **counts,
        "per_entity": {kind: dict(by_entity[kind]) for kind in guard.entities},
        "wrong_type_same_field": [
            {"gold": gold_kind, "predicted": predicted_kind, "count": count}
            for (gold_kind, predicted_kind), count in sorted(wrong_type.items())
        ],
        "full_mask_coverage": (
            counts["fully_masked_fields"] / counts["target_fields"]
            if counts["target_fields"] else None
        ),
        "exact_type_rate": (
            counts["exact_type_fields"] / counts["target_fields"]
            if counts["target_fields"] else None
        ),
        "changed_nontarget_rate": (
            counts["changed_nontarget_fields"] / counts["nontarget_fields"]
            if counts["nontarget_fields"] else None
        ),
        "negative_record_change_rate": (
            counts["negative_records_with_changed_field"] / counts["negative_records"]
            if counts["negative_records"] else None
        ),
    }


def worker(path: Path, entities: list[str], threshold: float,
           use_ner: bool, ner_model: str, ner_threshold: float) -> dict:
    import ko_pii_guard
    from ko_pii_guard import KoreanPIIGuard

    cases = read_cases(path, set(entities))
    ner = None
    model = None
    if use_ner:
        from ko_pii_guard.ner import MODELS, KoreanNER

        ner = KoreanNER.from_pretrained(model=ner_model, score_threshold=ner_threshold)
        model = {"id": MODELS[ner_model][0], "revision": MODELS[ner_model][1]}
    guard = KoreanPIIGuard(entities=entities, score_threshold=threshold, ner=ner)
    return {"loaded_source": str(Path(ko_pii_guard.__file__).resolve()),
            "ner_model": model, "records": len(cases), **evaluate(cases, guard)}


def measure(snapshot: Path, input_path: Path, entities: list[str], threshold: float,
            use_ner: bool, ner_model: str, ner_threshold: float) -> dict:
    environment = dict(os.environ, PYTHONPATH=str(snapshot / "src"))
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--input",
               str(input_path), "--threshold", str(threshold), "--ner-model", ner_model,
               "--ner-threshold", str(ner_threshold), "--entities", *entities]
    if use_ner:
        command.append("--ner")
    completed = subprocess.run(command, cwd=snapshot, env=environment,
                               check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout)
    if not Path(result.pop("loaded_source")).is_relative_to(snapshot):
        raise RuntimeError("Worker imported a different source tree")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--development-input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-ref", default="30bf0b0")
    parser.add_argument("--entities", nargs="+", default=None)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--ner", action="store_true")
    parser.add_argument("--ner-model", choices=("e5", "kcelectra"), default="e5")
    parser.add_argument("--ner-threshold", type=float, default=0.9)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    from ko_pii_guard.analyzer import DEFAULT_ENTITIES, SUPPORTED_ENTITIES

    entities = args.entities or list(DEFAULT_ENTITIES)
    if len(set(entities)) != len(entities) or not set(entities) <= set(SUPPORTED_ENTITIES):
        parser.error("--entities must be unique supported entity names")
    if not 0 <= args.threshold <= 1 or not 0 <= args.ner_threshold <= 1:
        parser.error("thresholds must be between 0 and 1")
    if args.ner and not {"KR_NAME", "KR_ADDRESS"}.intersection(entities):
        parser.error("--ner requires KR_NAME or KR_ADDRESS in --entities")
    if not args.ner and args.ner_model != "e5":
        parser.error("--ner-model kcelectra requires --ner")
    input_path = args.input.resolve(strict=True)
    development_path = (args.development_input.resolve(strict=True)
                        if args.development_input else None)
    if development_path == input_path:
        parser.error("--development-input must differ from --input")
    output_path = args.output.absolute() if args.output else None
    if (output_path and (output_path.is_symlink() or output_path.is_dir()
                         or output_path == input_path
                         or output_path == development_path
                         or (output_path.exists() and any(
                             os.path.samefile(output_path, source)
                             for source in (input_path, development_path) if source
                         )))):
        parser.error("--output must differ from inputs and cannot be a symlink or directory")
    if args.worker:
        print(json.dumps(worker(input_path, entities, args.threshold, args.ner,
                                args.ner_model, args.ner_threshold)))
        return
    with tempfile.TemporaryDirectory(prefix="ko-pii-structured-eval-") as directory:
        base = Path(directory)
        frozen = base / "records.jsonl"
        input_hash = freeze_input(input_path, frozen)
        cases = read_cases(frozen, set(entities))
        development_hash = None
        development_records = None
        if development_path:
            frozen_development = base / "development.jsonl"
            development_hash = freeze_input(development_path, frozen_development)
            development = read_cases(frozen_development, set(entities))
            check_split_overlap(cases, development)
            development_records = len(development)
        if args.validate_only:
            counts = Counter(kind for case in cases for kind in case["gold"].values())
            report = {"validation_only": True, "corpus_sha256": input_hash,
                      "development_corpus_sha256": development_hash,
                      "records": len(cases), "target_fields": sum(counts.values()),
                      "development_records": development_records,
                      "targets_by_entity": {kind: counts[kind] for kind in entities},
                      "negative_records": sum(not case["gold"] for case in cases),
                      "hard_negative_fields": sum(len(case["negative"]) for case in cases),
                      "ignored_fields": sum(len(case["ignore"]) for case in cases)}
        else:
            baseline_commit = subprocess.run(
                ["git", "rev-parse", f"{args.baseline_ref}^{{commit}}"], cwd=ROOT,
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            current_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT,
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            score_hash = scorer_hash()
            working_hash = source_hash(ROOT)
            baseline_root, current_root = base / "baseline", base / "current"
            archive_revision(baseline_commit, baseline_root)
            shutil.copytree(ROOT / "src" / "ko_pii_guard", current_root / "src" / "ko_pii_guard",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            baseline_hash = source_hash(baseline_root)
            current_hash = source_hash(current_root)
            if current_hash != working_hash or source_hash(ROOT) != working_hash:
                raise RuntimeError("Working source changed during snapshot")
            if current_commit == baseline_commit:
                head_hash = baseline_hash
            else:
                head_root = base / "head"
                archive_revision(current_commit, head_root)
                head_hash = source_hash(head_root)
            baseline = measure(baseline_root, frozen, entities, args.threshold,
                               args.ner, args.ner_model, args.ner_threshold)
            current = measure(current_root, frozen, entities, args.threshold,
                              args.ner, args.ner_model, args.ner_threshold)
            if scorer_hash() != score_hash:
                raise RuntimeError("Scoring scripts changed during evaluation")
            if subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                capture_output=True, text=True,
            ).stdout.strip() != current_commit:
                raise RuntimeError("HEAD changed during evaluation")
            if baseline["ner_model"] != current["ner_model"]:
                raise RuntimeError("Workers used different NER revisions")
            report = {
                "baseline_commit": baseline_commit, "current_commit": current_commit,
                "baseline_source_sha256": baseline_hash,
                "current_source_sha256": current_hash,
                "current_source_has_uncommitted_changes": current_hash != head_hash,
                "scorer_sha256": score_hash, "corpus_sha256": input_hash,
                "development_corpus_sha256": development_hash,
                "entities": entities, "threshold": args.threshold,
                "ner_enabled": args.ner,
                "ner_model_key": args.ner_model if args.ner else None,
                "ner_threshold": args.ner_threshold if args.ner else None,
                "python": platform.python_version(),
                "dependencies": {name: version(name) for name in (
                    "presidio-analyzer", "presidio-anonymizer", "phonenumbers", "tldextract",
                    *(("torch", "transformers", "tokenizers", "safetensors") if args.ner else ()),
                )},
                "baseline": baseline, "current": current,
                "scope": "whole-field tag type and stars coverage; no timing",
            }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if output_path:
        write_report(output_path, rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
