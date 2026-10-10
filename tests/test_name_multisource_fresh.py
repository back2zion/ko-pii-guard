"""Fresh annotations stay sealed until live gates and text amendment are verified."""

import argparse
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("presidio_analyzer")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import evaluate_name_multisource_fresh as fresh
finally:
    sys.path[:] = previous_path


def write(path, content):
    path.write_text(json.dumps(content))
    return path


def fixture(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    source = write(tmp_path / "klue-ner-v1.1_dev.tsv", {})
    exclusion = write(tmp_path / "exclusion.json", [])
    write(checkpoint / "data-manifest.json", {"input_sha256": {
        str(exclusion): fresh.live.digest(exclusion)}})
    reserve = write(tmp_path / "reserve.json", {
        "exclusion_hashes_path": str(exclusion),
        "exclusion_hashes_sha256": fresh.live.digest(exclusion),
        "source_sha256": {str(source): fresh.live.digest(source)},
    })
    consumed = write(tmp_path / "consumed.json", {})
    decision, development = write(tmp_path / "decision.json", {}), write(tmp_path / "dev.json", {})
    dev_reports = [write(tmp_path / (domain + "-live.json"), {"domain": domain})
                   for domain in ("klue", "kdpii")]
    amendment_path = write(tmp_path / "amendment.json", {})
    cases = [dict(id="one-nsmc", text="홍길동 왔다", expected=[
        dict(entity="KR_NAME", start=0, end=3)]),
        dict(id="two-wikitree", text="임꺽정 왔다", expected=[
            dict(entity="KR_NAME", start=0, end=3)])]
    original_ids = ["removed", *[r["id"] for r in cases]]
    amendment = dict(created_at_utc="2026-10-10T00:00:00+00:00", reason="text duplicates",
                     frozen_sha256={str(source): fresh.live.digest(source)}, domains={
                         "klue": dict(original_selected_ids=original_ids,
                                      selected_ids=[r["id"] for r in cases],
                                      excluded_ids=["removed"], original_count=3,
                                      selected_count=2, excluded_count=1,
                                      text_sha256={r["id"]: fresh.audit.text_hashes(r["text"])
                                                   for r in cases})})
    policy = dict(algorithm="selective_whole_span", addition_threshold=1.,
                  removal_threshold=.9999, class_logit_correction=0.)
    args = argparse.Namespace(domain="klue", checkpoint=checkpoint, reserve=reserve,
                              amendment=amendment_path, decision=decision, development=development,
                              live_development=dev_reports, device="cpu",
                              output=tmp_path / "fresh.json")
    monkeypatch.setitem(fresh.live.CONSUMED_REPORTS, "klue", consumed)
    monkeypatch.setattr(fresh.live, "validate_selected_decision", lambda *a: (
        dict(id="fixed-policy", policy=policy),
        {"frozen_sha256": {str(consumed): fresh.live.digest(consumed)}}, {}))
    monkeypatch.setattr(fresh.live, "validate_live_reports", lambda *a: None)
    monkeypatch.setattr(fresh.audit, "validate_amendment", lambda *a: amendment)
    monkeypatch.setattr(fresh.audit, "source_paths", lambda *a: {"klue": source})
    monkeypatch.setattr(fresh.live, "load_ancestry", lambda *a: (set(), set(), {}))
    monkeypatch.setattr(fresh.live, "model_snapshot_hashes", lambda: {})
    monkeypatch.setattr(fresh.live, "read_cases", lambda *a: cases)
    monkeypatch.setattr(fresh.live, "resolve_factory", lambda *a: (
        lambda config: pytest.fail("Unexpected model loading"), "test:factory"))
    return args, amendment, cases


def test_rejected_selection_never_opens_amendment_or_gold(tmp_path, monkeypatch):
    args, _, _ = fixture(tmp_path, monkeypatch)

    def reject(*args):
        raise ValueError("not selected")

    monkeypatch.setattr(fresh.live, "validate_selected_decision", reject)
    monkeypatch.setattr(fresh.audit, "validate_amendment", lambda *a: pytest.fail("Text opened"))
    monkeypatch.setattr(fresh.live, "read_cases", lambda *a: pytest.fail("Gold opened"))
    with pytest.raises(ValueError, match="not selected"):
        fresh.run(args)


def test_failed_live_gate_blocks_amendment_and_annotation_access(tmp_path, monkeypatch):
    args, _, _ = fixture(tmp_path, monkeypatch)

    def reject(reports, identity):
        assert identity["reserve_sha256"] == fresh.live.digest(args.reserve)
        assert "amendment_sha256" not in identity
        raise ValueError("live failed")

    monkeypatch.setattr(fresh.live, "validate_live_reports", reject)
    monkeypatch.setattr(fresh.audit, "validate_amendment", lambda *a: pytest.fail("Text opened"))
    monkeypatch.setattr(fresh.live, "read_cases", lambda *a: pytest.fail("Gold opened"))
    with pytest.raises(ValueError, match="live failed"):
        fresh.run(args)


def test_text_only_ancestry_blocks_before_gold_or_model_loading(tmp_path, monkeypatch):
    args, amendment, _ = fixture(tmp_path, monkeypatch)
    value = amendment["domains"]["klue"]["text_sha256"]["one-nsmc"]["normalized_sha256"]
    monkeypatch.setattr(fresh.live, "load_ancestry", lambda *a: (set(), {value}, {}))
    monkeypatch.setattr(fresh.live, "read_cases", lambda *a: pytest.fail("Gold opened"))
    with pytest.raises(SystemExit, match="Ancestry overlap"):
        fresh.run(args)
    result = json.loads(args.output.read_text())
    assert result["passed"] is False
    assert result["ancestry_overlap"]["normalized_ids"] == ["one-nsmc"]
    assert result["evaluation_blocked"] == "ancestry_text_overlap_no_resampling"


@pytest.mark.parametrize("change", ["order", "text"])
def test_annotation_parser_must_match_amended_ids_and_text_hashes(tmp_path, monkeypatch, change):
    args, _, cases = fixture(tmp_path, monkeypatch)
    if change == "order":
        cases.reverse()
    else:
        cases[0]["text"] = "different original text"
    monkeypatch.setattr(fresh.live.KoreanNER, "from_pretrained",
                        lambda **kw: pytest.fail("Model opened"))
    with pytest.raises(ValueError, match="amended|text"):
        fresh.run(args)


def test_fixed_fresh_driver_uses_real_public_masks_and_pins_manifest_before_gold(
    tmp_path, monkeypatch,
):
    args, amendment, cases = fixture(tmp_path, monkeypatch)
    order = []

    def reader(domain, mode, path, identifiers):
        assert mode == "heldout"
        manifest = json.loads(args.output.with_suffix(".manifest.json").read_text())
        assert manifest["selected_ids"] == identifiers == [row["id"] for row in cases]
        assert manifest["original_selected_count"] == 3 and manifest["excluded_count"] == 1
        assert manifest["amendment_sha256"] == fresh.live.digest(args.amendment)
        assert manifest["selected_policy"]["removal_threshold"] == .9999
        assert order == []
        order.append("gold")
        return cases

    class NER:
        device = "cpu"

        def analyze(self, text):
            return [fresh.live.RecognizerResult("KR_NAME", 0, 3, .99)]

    def baseline(**kwargs):
        assert kwargs["score_threshold"] == .9
        order.append("baseline")
        return NER()

    def candidate(config):
        assert config["model_kind"] == "frozen"
        assert config["addition_threshold"] == 1. and config["removal_threshold"] == .9999
        order.append("candidate")
        return NER()

    monkeypatch.setattr(fresh.live, "read_cases", reader)
    monkeypatch.setattr(fresh.live.KoreanNER, "from_pretrained", baseline)
    monkeypatch.setattr(fresh.live, "resolve_factory", lambda *a: (candidate, "test:factory"))
    fresh.run(args)
    result = json.loads(args.output.read_text())
    assert order == ["gold", "baseline", "candidate"]
    assert result["passed"] is True
    assert result["scope"] == "multisource_live_heldout"
    assert result["selected_ids"] == amendment["domains"]["klue"]["selected_ids"]
    assert result["source_changed_during_run"] is False
    assert result["sources"]["nsmc"]["metrics"]["fully_covered_names"] == 1
    manifest = Path(result["pre_inference_manifest_path"])
    assert result["pre_inference_manifest_sha256"] == fresh.live.digest(manifest)
    assert result["frozen_sha256"][str(manifest)] == fresh.live.digest(manifest)
