"""Accept repeated identical KDPII annotations without discarding source records.

An annotation instance ID is metadata. Equal (begin, end, label, form) instances
express the same span event and paint the same BIO labels. Only those repetitions
are collapsed for the frozen strict converter; all original instances are kept.
Different labels, partial overlaps, surfaces, and BIO inconsistencies still fail.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from fetch_kdpii_evaluation import convert_record

VERSION = "identical_annotation_duplicates_v1"


def _annotation(entity):
    return dict(label=entity["label"], start=entity["begin"], end=entity["end"],
                form=entity["form"])


def _ordered(annotations):
    return sorted(annotations, key=lambda row: (row["start"], row["end"], row["label"]))


def convert_record_lossless(record):
    original = record["PII_set"]
    if not isinstance(original, list):
        raise ValueError("Expected the original PII annotation list")
    unique, grouped = [], {}
    for index, entity in enumerate(original):
        start, end, label, form = (entity[key] for key in ("begin", "end", "label", "form"))
        if (type(start) is not int or type(end) is not int
                or not isinstance(label, str) or not isinstance(form, str)):
            raise ValueError("Invalid annotation span, label, or form")
        signature = start, end, label, form
        if signature not in grouped:
            unique.append(entity)
            grouped[signature] = []
        grouped[signature].append(index)
    # This validates every unique span/surface, the character array, and complete
    # original BIO labels. Unequal overlapping spans remain a strict error.
    case = convert_record(dict(record, PII_set=unique))
    case["source_annotations"] = _ordered([_annotation(entity) for entity in original])
    case["original_PII_set"] = copy.deepcopy(original)
    case["duplicate_annotations"] = [dict(label=label, start=start, end=end, indices=indices)
                                     for (start, end, label, _), indices in grouped.items()
                                     if len(indices) > 1]
    return case


def validate_cases_lossless(cases, selected_ids):
    if [case["id"] for case in cases] != selected_ids or len(set(selected_ids)) != len(cases):
        raise ValueError("Cases differ from the complete fixed selected ID order")
    for case in cases:
        original = _ordered([_annotation(entity) for entity in case["original_PII_set"]])
        if case["source_annotations"] != original:
            raise ValueError("Every original source annotation instance must be preserved")
        for field, labels in (("expected", {"PS_NAME"}),
                              ("alternate_expected", {"PS_NAME", "PS_NICKNAME"})):
            expected = sorted({(row["start"], row["end"]) for row in original
                               if row["label"] in labels})
            actual = sorted((row["start"], row["end"]) for row in case[field]
                            if row["entity"] == "KR_NAME")
            if actual != expected or len(actual) != len(case[field]):
                raise ValueError("Name gold must equal unique original source span events")
            if any(not 0 <= start < end <= len(case["text"]) for start, end in actual):
                raise ValueError("Gold offsets must use original sentence coordinates")


def read_cases_lossless(path, selected_ids):
    wanted = set(selected_ids)
    if len(wanted) != len(selected_ids):
        raise ValueError("Duplicate selected IDs")
    records = json.loads(Path(path).read_text())
    rows = [row for row in records if row["sent_idx"] in wanted]
    selected = {row["sent_idx"]: row for row in rows}
    if set(selected) != wanted or len(rows) != len(selected):
        raise ValueError("Missing or duplicated selected KDPII IDs; no replacement permitted")
    cases = [convert_record_lossless(selected[identifier]) for identifier in selected_ids]
    validate_cases_lossless(cases, selected_ids)
    duplicates = [dict(id=case["id"], duplicates=case["duplicate_annotations"])
                  for case in cases if case["duplicate_annotations"]]
    return cases, duplicates
