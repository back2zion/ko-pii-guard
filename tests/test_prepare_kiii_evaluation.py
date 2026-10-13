"""The external finance sample must be selected without looking at labels."""

import json

from scripts.prepare_kiii_evaluation import convert, selected_rows


def test_selection_is_stratified_and_label_blind(tmp_path):
    rows = [
        {"doc_id": f"{kind}-{level}-{length}-{variant}", "doc_type": f"kind-{kind}",
         "variation_level": level, "context_len_bucket": length,
         "text": "sample", "spans": [{"category": "rrn"}] if variant else []}
        for kind in range(10) for level in ("T0", "T3")
        for length in ("1k", "4k", "16k") for variant in range(2)
    ]
    path = tmp_path / "source.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    selected = selected_rows(path)
    assert len(selected) == 60
    assert len(selected_rows(path, ("1k",))) == 20
    first_ids = {row["doc_id"] for row in selected}
    for row in rows:
        row["spans"] = [] if row["spans"] else [{"category": "rrn"}]
    path.write_text("\n".join(json.dumps(row) for row in reversed(rows)) + "\n")
    assert {row["doc_id"] for row in selected_rows(path)} == first_ids


def test_only_supported_gold_and_specific_unsupported_identifiers_are_scored():
    row = {"doc_id": "one", "text": "0123456789abcdefghij",
           "spans": [
               {"category": "rrn", "start": 0, "end": 10, "surface": "0123456789"},
               {"category": "customer_id", "start": 10, "end": 15, "surface": "abcde"},
               {"category": "consultation_content", "start": 15, "end": 20,
                "surface": "fghij"},
           ]}
    case = convert(row, include_name_address=False)
    assert case["entities"] == [{"type": "KR_RRN", "start": 0, "end": 10}]
    assert case["ignore"] == [{"start": 10, "end": 15}]
    name_only = convert(row, include_name_address=False, only_name_address=True)
    assert name_only["entities"] == []
    assert {tuple(item.values()) for item in name_only["ignore"]} == {(0, 10), (10, 15)}
