"""All retired v3 non-person sentences must remain unmasked."""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    c for line in (ROOT / "benchmarks/data/name_context_v3.jsonl").read_text().splitlines()
    if (c := json.loads(line))["split"] == "evaluation" and not c["expected"]
]


def test_retired_v3_negative_cases_are_complete():
    assert len(CASES) == 40
    report = json.loads((ROOT / "artifacts/name-context-v3/training_report.json").read_text())
    failed = set(report["evaluation_candidate"]["error_ids"]["negative_false_positive"])
    assert len(failed) == 12
    assert failed <= {c["id"] for c in CASES}


@pytest.fixture(scope="module")
def guard():
    torch = pytest.importorskip("torch")
    from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard
    from ko_pii_guard.ner import KoreanNER

    torch.set_num_threads(2)
    return KoreanPIIGuard(
        entities=SUPPORTED_ENTITIES,
        ner=KoreanNER.from_pretrained(name_context_path=os.environ.get(
            "KO_PII_TEST_NAME_CONTEXT_PATH", str(ROOT / "artifacts/name-context-v6-filter-retry")
        )),
    )


@pytest.mark.skipif(os.environ.get("KO_PII_TEST_NAME_CONTEXT") != "1",
                    reason="requires cached optional NER")
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_non_person_sentence_has_no_findings_or_replacement(guard, case):
    assert guard.analyze(case["text"]) == []
    assert guard.mask(case["text"]) == case["text"]
    assert guard.mask(case["text"], style="stars") == case["text"]
