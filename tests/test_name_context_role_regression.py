"""Every previously correct person sentence remains an exact masking contract."""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = [json.loads(line) for line in (
    ROOT / "benchmarks/data/name_context_v5.jsonl"
).read_text().splitlines() if json.loads(line)["id"] == "v5-evaluation-person-02-010"]
assert len(CASES) == 1


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
def test_previous_correct_person_spans_and_masks_are_preserved(guard, case):
    text = case["text"]
    assert [(f.entity, f.start, f.end) for f in guard.analyze(text)] == [
        (e["entity"], e["start"], e["end"]) for e in case["expected"]
    ]
    stars = list(text)
    for span in case["expected"]:
        stars[span["start"]:span["end"]] = "*" * (span["end"] - span["start"])
    assert guard.mask(text, style="stars") == "".join(stars)
