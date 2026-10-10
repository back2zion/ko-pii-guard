"""A non-person title must not suppress the genuine author in the same sentence."""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    c for line in (ROOT / "benchmarks/data/name_context_v4.jsonl").read_text().splitlines()
    if (c := json.loads(line))["id"].startswith("v4-train-person-05-")
]


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
def test_title_is_preserved_but_author_is_fully_masked(guard, case):
    text = case["text"]
    assert len(case["expected"]) == 1
    span = case["expected"][0]
    assert [(f.entity, f.start, f.end) for f in guard.analyze(text)] == [
        ("KR_NAME", span["start"], span["end"])
    ]
    assert guard.mask(text, style="stars") == (
        text[:span["start"]] + "*" * (span["end"] - span["start"]) + text[span["end"]:]
    )
