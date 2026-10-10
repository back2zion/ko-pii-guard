"""A false-positive reduction never compensates for a newly lost person span."""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
previous_path = sys.path[:]
sys.path.insert(0, str(ROOT / "benchmarks"))
try:
    GATES = [runpy.run_path(str(ROOT / "benchmarks" / name))["nonregression_gate"]
             for name in ("name_context_v5_benchmark.py", "name_context_v6_benchmark.py")]
finally:
    sys.path[:] = previous_path

def GATE(*args):
    reports = [gate(*args) for gate in GATES]
    assert reports[0] == reports[1]
    return reports[1]


CASES = [dict(id="person", text="이름 갑 을", expected=[
    dict(entity="KR_NAME", start=3, end=4), dict(entity="KR_NAME", start=5, end=6)
]), dict(id="negative", text="보통명사", expected=[])]
FIRST, SECOND, FALSE = ("KR_NAME", 3, 4), ("KR_NAME", 5, 6), ("KR_NAME", 0, 2)


def test_reducing_negative_errors_cannot_hide_a_lost_person():
    before = {"이름 갑 을": {FIRST}, "보통명사": {FALSE}}
    after = {"이름 갑 을": {SECOND}, "보통명사": set()}
    report = GATE(CASES, before, after)
    assert report["current_false_positive_spans"] < report["previous_false_positive_spans"]
    assert report["lost_correct_spans"] == [("person", *FIRST)]
    assert not report["nonregression_passed"]


def test_even_fewer_total_false_predictions_cannot_introduce_a_new_false_span():
    before = {"이름 갑 을": {FIRST, FALSE}, "보통명사": {FALSE}}
    after = {"이름 갑 을": {FIRST, SECOND}, "보통명사": {("KR_NAME", 2, 4)}}
    report = GATE(CASES, before, after)
    assert report["current_false_positive_spans"] < report["previous_false_positive_spans"]
    assert report["introduced_false_positive_spans"] == [("negative", "KR_NAME", 2, 4)]
    assert not report["nonregression_passed"]


def test_recovered_names_and_removed_false_spans_without_new_errors_pass():
    before = {"이름 갑 을": {FIRST}, "보통명사": {FALSE}}
    after = {"이름 갑 을": {FIRST, SECOND}, "보통명사": set()}
    report = GATE(CASES, before, after)
    assert report["nonregression_passed"]
    assert report["lost_correct_spans"] == report["introduced_false_positive_spans"] == []
