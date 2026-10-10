"""The parser compatibility continuation keeps the original sealed decision."""

import argparse
import copy
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("presidio_analyzer")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import evaluate_name_multisource_fresh_lossless as fresh
finally:
    sys.path[:] = previous_path


def write(path, value):
    path.write_text(json.dumps(value))
    return path


def fixture(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    text = "팀장 홍길동"
    rows = [dict(sent_idx="one", sentence=text, sent_seq=list(text),
                 labelling_seq=["B-CV_POSITION", "I-CV_POSITION", "O", "B-PS_NAME",
                                "I-PS_NAME", "I-PS_NAME"],
                 PII_set=[dict(id=0, begin=0, end=2, label="CV_POSITION", form="팀장"),
                          dict(id=1, begin=3, end=6, label="PS_NAME", form="홍길동"),
                          dict(id=2, begin=0, end=2, label="CV_POSITION", form="팀장")])]
    source = write(tmp_path / "test.json", rows)
    exclusion = write(tmp_path / "exclusion.json", [])
    write(checkpoint / "data-manifest.json", {"input_sha256": {
        str(exclusion): fresh.live.digest(exclusion)}})
    reserve = write(tmp_path / "reserve.json", {
        "exclusion_hashes_path": str(exclusion),
        "exclusion_hashes_sha256": fresh.live.digest(exclusion),
        "source_sha256": {str(source): fresh.live.digest(source)}})
    consumed = write(tmp_path / "consumed.json", {})
    decision, development = write(tmp_path / "decision.json", {}), write(tmp_path / "dev.json", {})
    dev_reports = [write(tmp_path / (domain + "-live.json"), {"domain": domain})
                   for domain in ("klue", "kdpii")]
    amendment_path = write(tmp_path / "amendment.json", {})
    population = dict(original_selected_ids=["removed", "one"], selected_ids=["one"],
                      excluded_ids=["removed"], original_count=2, selected_count=1,
                      excluded_count=1, text_sha256={"one": fresh.audit.text_hashes(text)})
    amendment = dict(created_at_utc="2026-10-10T00:00:00+00:00", reason="text duplicates",
                     frozen_sha256={str(source): fresh.live.digest(source)},
                     domains={"kdpii": population})
    policy = dict(algorithm="selective_whole_span", addition_threshold=1.,
                  removal_threshold=.9999, class_logit_correction=0.)
    identity = dict(decision_sha256=fresh.live.digest(decision),
                    development_report_sha256=fresh.live.digest(development), artifact_sha256={},
                    selected_policy=policy, reserve_sha256=fresh.live.digest(reserve),
                    device="cpu", runtime_signature=fresh.live.runtime_signature())
    original_manifest = write(tmp_path / "aborted.manifest.json", dict(
        scope="multisource_live_heldout", domain="kdpii", **identity,
        selected_ids=population["selected_ids"], original_selected_ids=population[
            "original_selected_ids"], excluded_ids=population["excluded_ids"],
        amendment_sha256=fresh.live.digest(amendment_path),
        frozen_sha256={str(source): fresh.live.digest(source),
                       str(fresh.ORIGINAL_DRIVER): fresh.live.digest(fresh.ORIGINAL_DRIVER)}))
    args = argparse.Namespace(domain="kdpii", checkpoint=checkpoint, reserve=reserve,
                              amendment=amendment_path, decision=decision, development=development,
                              live_development=dev_reports, device="cpu",
                              original_failure_manifest=original_manifest,
                              output=tmp_path / "fresh-lossless.json")
    monkeypatch.setitem(fresh.live.CONSUMED_REPORTS, "kdpii", consumed)
    monkeypatch.setattr(fresh.live, "validate_selected_decision", lambda *a: (
        dict(id="fixed-policy", policy=policy),
        {"frozen_sha256": {str(consumed): fresh.live.digest(consumed)}}, {}))
    monkeypatch.setattr(fresh.live, "validate_live_reports", lambda *a: None)
    monkeypatch.setattr(fresh.audit, "validate_amendment", lambda *a: amendment)
    monkeypatch.setattr(fresh.audit, "source_paths", lambda *a: {"kdpii": source})
    monkeypatch.setattr(fresh.live, "load_ancestry", lambda *a: (set(), set(), {}))
    monkeypatch.setattr(fresh.live, "model_snapshot_hashes", lambda: {})
    monkeypatch.setattr(fresh.live, "resolve_factory", lambda *a: (
        lambda config: pytest.fail("Unexpected model loading"), "test:factory"))
    return args, identity, amendment, population


def test_continuation_rejects_changed_original_policy_before_annotation_access(
    tmp_path, monkeypatch,
):
    args, _, _, _ = fixture(tmp_path, monkeypatch)
    original = json.loads(args.original_failure_manifest.read_text())
    original["selected_policy"]["removal_threshold"] = .99
    write(args.original_failure_manifest, original)
    monkeypatch.setattr(fresh.lossless, "read_cases_lossless",
                        lambda *a: pytest.fail("Annotations opened"))
    with pytest.raises(ValueError, match="original|continuation"):
        fresh.run(args)


def test_continuation_rejects_changed_original_driver_pin(tmp_path, monkeypatch):
    args, identity, amendment, population = fixture(tmp_path, monkeypatch)
    original = json.loads(args.original_failure_manifest.read_text())
    original["frozen_sha256"][str(fresh.ORIGINAL_DRIVER)] = "0" * 64
    write(args.original_failure_manifest, original)
    with pytest.raises(ValueError, match="changed|original"):
        fresh.validate_parser_continuation(args.original_failure_manifest, identity,
                                           args.amendment, population)


def test_live_validation_failure_precedes_parser_continuation(tmp_path, monkeypatch):
    args, _, _, _ = fixture(tmp_path, monkeypatch)

    def reject(*args):
        raise ValueError("live failed")

    monkeypatch.setattr(fresh.live, "validate_live_reports", reject)
    monkeypatch.setattr(fresh.lossless, "read_cases_lossless",
                        lambda *a: pytest.fail("Annotations opened"))
    with pytest.raises(ValueError, match="live failed"):
        fresh.run(args)


def test_all_original_cases_and_duplicate_metadata_are_frozen_before_actual_public_inference(
    tmp_path, monkeypatch,
):
    args, _, _, _ = fixture(tmp_path, monkeypatch)
    order = []
    real_reader = fresh.lossless.read_cases_lossless

    def reader(*a):
        order.append("validated_annotations")
        return real_reader(*a)

    class NER:
        device = "cpu"

        def analyze(self, text):
            return [fresh.live.RecognizerResult("KR_NAME", 3, 6, .99)]

    def baseline(**kwargs):
        manifest = json.loads(args.output.with_suffix(".manifest.json").read_text())
        parser = manifest["parser_compatibility"]
        assert parser["duplicate_records"] == [dict(id="one", duplicates=[
            dict(label="CV_POSITION", start=0, end=2, indices=[0, 2])])]
        assert parser["dropped_records"] == 0 and parser["selected_count"] == 1
        assert parser["original_failure_manifest_sha256"] == fresh.live.digest(
            args.original_failure_manifest)
        assert manifest["selected_policy"]["removal_threshold"] == .9999
        order.append("baseline")
        return NER()

    def candidate(config):
        assert config["model_kind"] == "frozen" and config["addition_threshold"] == 1.
        order.append("candidate")
        return NER()

    monkeypatch.setattr(fresh.lossless, "read_cases_lossless", reader)
    monkeypatch.setattr(fresh.live.KoreanNER, "from_pretrained", baseline)
    monkeypatch.setattr(fresh.live, "resolve_factory", lambda *a: (candidate, "test:factory"))
    fresh.run(args)
    result = json.loads(args.output.read_text())
    assert order == ["validated_annotations", "baseline", "candidate"]
    assert result["passed"] is True and result["selected_ids"] == ["one"]
    assert result["source_changed_during_run"] is False
    for field in ("helper", "reference_converter", "original_driver", "driver",
                  "original_failure_manifest"):
        parser = result["parser_compatibility"]
        assert result["frozen_sha256"][parser[field + "_path"]] == parser[field + "_sha256"]
    assert result["sources"]["kdpii/PS_NAME"]["metrics"]["gold_names"] == 1
    mutated = copy.deepcopy(result)
    assert mutated["parser_compatibility"]["gold_policy_changed"] is False
