"""Amend a fixed evaluation reserve using text identity only, never annotations.

KLUE reads its character column and skips its tag column. KDPII uses the standard
JSON syntax parser, then accesses only sent_idx and sentence; annotation values
are not interpreted or used. No model, gold conversion, or predictions are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ko_pii_guard import normalization

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("klue", "kdpii")
ORIGINAL_COUNTS = {"klue": 1000, "kdpii": 2000}
SCOPE = "multisource_reserve_text_amendment"
REASON = ("Exclude previously consumed exact/runtime-normalized text and retain only the "
          "first remaining runtime-normalized text in original KLUE-then-KDPII reserve order; "
          "no replacement sampling, annotation inspection, or model-policy change")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def text_hashes(text):
    return dict(exact_sha256=hashlib.sha256(text.encode()).hexdigest(),
                normalized_sha256=hashlib.sha256(
                    normalization.normalize_text(text).text.encode()).hexdigest())


def read_klue_texts(path, wanted_ids):
    """Read selected character columns only; header sentence and tags are ignored."""
    wanted, result, current, chars = set(wanted_ids), {}, None, []

    def finish():
        if current in wanted:
            if current in result:
                raise ValueError("Duplicate target KLUE ID")
            result[current] = "".join(chars)

    with Path(path).open() as stream:
        for raw in stream:
            line = raw.rstrip("\r\n")
            if line.startswith("## klue-ner-"):
                finish()
                current, chars = line[3:].split("\t", 1)[0].strip(), []
            elif current in wanted and line:
                if "\t" not in line:
                    raise ValueError("Malformed KLUE character-column row")
                chars.append(line.rsplit("\t", 1)[0])
    finish()
    if set(result) != wanted:
        raise ValueError("Selected KLUE text IDs are absent; no replacement permitted")
    return result


def read_kdpii_texts(path, wanted_ids):
    """Parse JSON syntax but access only ID and sentence, never annotation fields."""
    wanted, result = set(wanted_ids), {}
    records = json.loads(Path(path).read_text())
    if not isinstance(records, list):
        raise ValueError("Expected the pinned KDPII JSON record array")
    for row in records:
        identifier = row["sent_idx"]
        if identifier in wanted:
            if identifier in result:
                raise ValueError("Duplicate target KDPII ID")
            if not isinstance(row["sentence"], str):
                raise ValueError("KDPII sentence is not text")
            result[identifier] = row["sentence"]
    if set(result) != wanted:
        raise ValueError("Selected KDPII text IDs are absent; no replacement permitted")
    return result


def select_text_unique(reserve, texts):
    """Use the consumed union across domains, then preserve global reserve order."""
    fingerprints = {domain: {identifier: text_hashes(text) for identifier, text in rows.items()}
                    for domain, rows in texts.items()}
    consumed_exact, consumed_normalized = set(), set()
    for domain in DOMAINS:
        for identifier in reserve[domain]["consumed_ids"]:
            hashes = fingerprints[domain][identifier]
            consumed_exact.add(hashes["exact_sha256"])
            consumed_normalized.add(hashes["normalized_sha256"])
    seen_exact, seen_normalized, result = {}, {}, {}
    for domain in DOMAINS:
        original, selected, excluded = reserve[domain]["selected_ids"], [], []
        for identifier in original:
            hashes = fingerprints[domain][identifier]
            exact, normalized = hashes["exact_sha256"], hashes["normalized_sha256"]
            reasons = []
            if exact in consumed_exact:
                reasons.append("consumed_exact_duplicate")
            if normalized in consumed_normalized:
                reasons.append("consumed_normalized_duplicate")
            if exact in seen_exact:
                reasons.append("reserve_exact_duplicate")
            if normalized in seen_normalized:
                reasons.append("reserve_normalized_duplicate")
            if reasons:
                excluded.append(dict(id=identifier, reasons=reasons, **hashes,
                                     duplicate_of=seen_normalized.get(normalized)))
            else:
                selected.append(identifier)
                seen_exact[exact] = seen_normalized[normalized] = dict(domain=domain, id=identifier)
        result[domain] = dict(
            original_selected_ids=original, selected_ids=selected,
            excluded_ids=[row["id"] for row in excluded], original_count=len(original),
            selected_count=len(selected), excluded_count=len(excluded),
            consumed_count=len(reserve[domain]["consumed_ids"]), exclusions=excluded,
            text_sha256=fingerprints[domain],
        )
    return result


def source_paths(reserve):
    names = {"klue": "klue-ner-v1.1_dev.tsv", "kdpii": "test.json"}
    result = {}
    pins = reserve.get("source_sha256")
    if not isinstance(pins, dict) or not pins:
        raise ValueError("Reserve requires source SHA256 pins")
    for domain, name in names.items():
        matches = [Path(path) for path in pins if Path(path).name == name]
        if len(matches) != 1:
            raise ValueError("Require one pinned source for each domain")
        result[domain] = matches[0]
    return result


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if not Path(path).is_file() or digest(path) != expected:
            raise ValueError(f"Text audit input changed or is missing: {path}")


def build_amendment(reserve_path):
    reserve_path = Path(reserve_path).resolve()
    reserve_hash = digest(reserve_path)
    reserve = json.loads(reserve_path.read_text())
    if (reserve.get("scope") != "v2_evaluation_reserve_before_any_new_training"
            or reserve.get("test_labels_read") is not False):
        raise ValueError("Require the original ID-only reserve")
    for domain, count in ORIGINAL_COUNTS.items():
        selected, consumed = reserve[domain]["selected_ids"], reserve[domain]["consumed_ids"]
        if (not isinstance(selected, list) or not isinstance(consumed, list)
                or any(not isinstance(x, str) or not x for x in [*selected, *consumed])
                or len(selected) != count or len(set(selected)) != count
                or len(set(consumed)) != len(consumed) or set(selected) & set(consumed)):
            raise ValueError("Invalid original reserve population or consumed ID overlap")
    sources = source_paths(reserve)
    hashes = {**reserve["source_sha256"], str(reserve_path): reserve_hash,
              str(Path(__file__).resolve()): digest(__file__),
              str(Path(normalization.__file__).resolve()): digest(normalization.__file__)}
    verify_hashes(hashes)
    texts = {domain: reader(sources[domain], [*reserve[domain]["consumed_ids"],
                                           *reserve[domain]["selected_ids"]])
             for domain, reader in (("klue", read_klue_texts), ("kdpii", read_kdpii_texts))}
    domains = select_text_unique(reserve, texts)
    verify_hashes(hashes)
    return dict(
        scope=SCOPE, created_at_utc=datetime.now(timezone.utc).isoformat(), reason=REASON,
        original_reserve_path=str(reserve_path), original_reserve_sha256=reserve_hash,
        source_sha256=reserve["source_sha256"], frozen_sha256=hashes, domains=domains,
        domain_order=list(DOMAINS), normalization="runtime_normalize_text",
        consumed_comparison="Union of both domains before any reserve retention",
        labels_interpreted=False, predictions_read=False, model_inference=False,
        replacement_sampling=False, policy_changed=False, weights_changed=False,
        parsing_policy="Standard JSON syntax parsed; only sent_idx/sentence accessed. "
                       "KLUE character columns only; tags and header sentences ignored.",
        source_changed_during_run=False,
    )


def validate_amendment(amendment_path, reserve_path):
    """Recompute every deterministic field and reject changed inputs or amended IDs."""
    amendment_path = Path(amendment_path)
    before = digest(amendment_path)
    actual = json.loads(amendment_path.read_text())
    if not isinstance(actual, dict) or not isinstance(actual.get("frozen_sha256"), dict):
        raise ValueError("Require a complete text amendment report")
    verify_hashes(actual["frozen_sha256"])
    stamp = datetime.fromisoformat(actual.get("created_at_utc", ""))
    if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError("Require an explicit UTC amendment timestamp")
    expected = build_amendment(reserve_path)
    expected["created_at_utc"] = actual["created_at_utc"]
    if actual != expected or digest(amendment_path) != before:
        raise ValueError("Amendment differs from the recomputed text-only exclusion")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reserve", type=Path,
                        default=ROOT / "benchmarks/results/name-multisource-v2-reserve.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Require an unused amendment output path")
    result = build_amendment(args.reserve)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({domain: {key: row[key] for key in (
        "original_count", "selected_count", "excluded_count", "excluded_ids")}
        for domain, row in result["domains"].items()}))


if __name__ == "__main__":
    main()
