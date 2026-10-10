"""Previously held-out v2 errors are now named development regressions."""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = json.loads((ROOT / "benchmarks/results/name-context-v2-holdout.json").read_text())
FAILED_IDS = set(REPORT["profiles"]["v2"]["error_ids"]["exact_span"])
CASES = [
    case
    for line in (ROOT / "benchmarks/data/name_context_v2.jsonl").read_text().splitlines()
    if (case := json.loads(line))["id"] in FAILED_IDS
]


@pytest.fixture(scope="module")
def guard():
    torch = pytest.importorskip("torch")
    from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard
    from ko_pii_guard.ner import KoreanNER

    torch.set_num_threads(2)
    return KoreanPIIGuard(
        entities=SUPPORTED_ENTITIES,
        ner=KoreanNER.from_pretrained(
            name_context_path=os.environ.get(
                "KO_PII_TEST_NAME_CONTEXT_PATH", str(ROOT / "artifacts/name-context-v3")
            )
        ),
    )


def test_all_v2_failed_sentence_ids_are_preserved():
    assert len(CASES) == len(FAILED_IDS) == 26
    assert {case["id"] for case in CASES} == FAILED_IDS
    assert all(case["split"] == "evaluation" for case in CASES)


@pytest.mark.skipif(
    os.environ.get("KO_PII_TEST_NAME_CONTEXT") != "1", reason="requires cached optional NER"
)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_previous_v2_errors_have_exact_spans_and_masking(guard, case):
    text = case["text"]
    findings = guard.analyze(text)
    assert [(f.entity, f.start, f.end) for f in findings] == [
        (e["entity"], e["start"], e["end"]) for e in case["expected"]
    ]
    assert all(f.text == text[f.start : f.end] for f in findings)
    stars = list(text)
    for span in case["expected"]:
        stars[span["start"] : span["end"]] = "*" * (span["end"] - span["start"])
    assert guard.mask(text, style="stars") == "".join(stars)
