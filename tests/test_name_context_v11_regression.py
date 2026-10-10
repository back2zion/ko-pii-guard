"""A better aggregate score must never hide loss of the correctly detected 공감 span."""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASE = next(json.loads(line) for line in (
    ROOT / "benchmarks/data/name_context_v10_regressions.jsonl"
).read_text().splitlines() if json.loads(line)["id"] == "v10-evaluation-person-02-012")


@pytest.mark.skipif(os.environ.get("KO_PII_TEST_NAME_CONTEXT") != "1",
                    reason="requires cached optional NER")
def test_previously_correct_span_and_its_mask_survive_filter_refinement():
    torch = pytest.importorskip("torch")
    from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard
    from ko_pii_guard.ner import KoreanNER

    torch.set_num_threads(2)
    guard = KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=KoreanNER.from_pretrained(
        name_context_path=os.environ.get("KO_PII_TEST_NAME_CONTEXT_PATH",
                                         str(ROOT / "artifacts/name-context-v11"))
    ))
    assert ("KR_NAME", 11, 13) in [(f.entity, f.start, f.end) for f in guard.analyze(CASE["text"])]
    assert guard.mask(CASE["text"], style="stars")[11:13] == "**"
