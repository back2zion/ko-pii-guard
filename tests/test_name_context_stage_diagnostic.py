"""Ensure attribution cannot blame a later stage for an earlier failure."""
import runpy
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")

ROOT = Path(__file__).resolve().parents[1]
previous_path = sys.path[:]
sys.path.insert(0, str(ROOT / "benchmarks"))
try:
    diagnose = runpy.run_path(
        str(ROOT / "benchmarks/name_context_stage_diagnostic.py")
    )["missing_stage"]
finally:
    sys.path[:] = previous_path

GOLD = ("KR_NAME", 4, 7)
WRONG_BOUNDARY = ("KR_NAME", 4, 8)


@pytest.mark.parametrize(("stages", "expected"), [
    ([{WRONG_BOUNDARY}] * 4, "no_exact_proposal"),
    ([{GOLD}, set(), set(), set()], "below_threshold"),
    ([{GOLD}, {GOLD}, set(), set()], "filtered_out"),
    ([{GOLD}, {GOLD}, {GOLD}, {WRONG_BOUNDARY}], "overlap_blocked"),
    ([{GOLD}] * 4, "downstream"),
])
def test_first_missing_exact_span_is_attributed_to_its_stage(stages, expected):
    assert diagnose(GOLD, *stages) == expected


def test_another_heads_exact_proposal_survives_union_attribution():
    first_head, second_head = {WRONG_BOUNDARY}, {GOLD}
    assert diagnose(GOLD, first_head | second_head, {GOLD}, {GOLD}, {GOLD}) == "downstream"
