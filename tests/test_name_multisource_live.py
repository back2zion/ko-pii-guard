"""Live evaluations cannot bypass selection, replay parity, or ancestry checks."""

import copy
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("presidio_analyzer")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import evaluate_name_multisource_live as live
    from name_multisource_selection import select_candidates
finally:
    sys.path[:] = previous_path


def write(path, value):
    path.write_text(json.dumps(value))
    return path


def score(tp=1):
    return dict(tp=tp, fp=0, fn=2-tp, gold_names=2, fully_covered_names=tp,
                unnecessary_masked_characters=0, negative_false_positive_sentences=0,
                f1=2*tp/(2*tp+2-tp), sentences=2, negative_sentences=0)


def gate():
    return dict(passed=True, lost_correct_spans=[], introduced_false_spans=[],
                newly_exposed_names=[], actual_stars_newly_exposed_names=[])


def selected_fixture(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "head.safetensors").write_bytes(b"test weights")
    write(checkpoint / "config.json", {"selected_epoch": 1, "threshold": .5,
                                       "model_id": live.MODEL_ID, "revision": live.MODEL_REVISION})
    write(checkpoint / "report.json", {"source_changed": False, "selected_epoch": 1,
                                       "threshold": .5, "history": [{"epoch": 1}]})
    write(checkpoint / "manifest.json", {"epochs": 1})
    write(checkpoint / "training-text-hashes.json", {
        "algorithm": "sha256_utf8_exact_text", "text_sha256": []})
    policy = {"cutoff": .5, "class_logit_correction": 0., "protect_anchors": True}
    domains = list(live.REQUIRED_SOURCES)
    report = dict(scope="multisource_development_only", source_changed_during_run=False,
                  baseline_replay_verified=True, required_sources=domains,
                  baselines={domain: score() for domain in domains},
                  candidates=[dict(id="chosen", policy=policy, sources={domain: dict(
                      metrics=score(2), regression_gate=gate()) for domain in domains})],
                  frozen_sha256={str(p.resolve()): live.digest(p) for p in checkpoint.iterdir()})
    development = write(tmp_path / "development.json", report)
    decision = select_candidates(report["baselines"], report["candidates"],
                                 required_sources=domains)
    decision.update(development_report_sha256=live.digest(development),
                    checker_sha256=live.digest(live.ROOT / "scripts/name_multisource_selection.py"))
    decision_path = write(tmp_path / "decision.json", decision)
    return checkpoint, development, decision_path, report, decision


def test_only_recomputed_unchanged_selected_policy_can_start(tmp_path):
    checkpoint, development, decision_path, _, decision = selected_fixture(tmp_path)
    chosen, _, _ = live.validate_selected_decision(decision_path, development, checkpoint)
    assert chosen["policy"] == decision["selected_policy"]
    altered = copy.deepcopy(decision)
    altered["selected_policy"]["cutoff"] = .1
    write(decision_path, altered)
    with pytest.raises(ValueError, match="decision"):
        live.validate_selected_decision(decision_path, development, checkpoint)
    write(decision_path, decision)
    (checkpoint / "head.safetensors").write_bytes(b"different weights")
    with pytest.raises(ValueError, match="changed|pinned"):
        live.validate_selected_decision(decision_path, development, checkpoint)


def test_rejected_decision_fails_before_source_or_model_access(tmp_path, monkeypatch):
    checkpoint, development, decision_path, _, decision = selected_fixture(tmp_path)
    write(decision_path, dict(decision, status="no_candidate"))
    monkeypatch.setattr(live, "read_cases", lambda *a, **k: pytest.fail("Opened source"))
    monkeypatch.setattr(live, "model_snapshot_hashes", lambda: pytest.fail("Opened model"))
    monkeypatch.setattr(sys, "argv", ["live", "--mode", "heldout", "--domain", "klue",
                                      "--decision", str(decision_path), "--development",
                                      str(development), "--checkpoint", str(checkpoint),
                                      "--output", str(tmp_path / "result.json")])
    with pytest.raises(ValueError, match="selected"):
        live.main()


def test_selected_policy_still_cannot_open_heldout_before_both_live_reports(tmp_path, monkeypatch):
    checkpoint, development, decision_path, _, _ = selected_fixture(tmp_path)
    reserve = write(tmp_path / "reserve.json", {"not_opened": True})
    monkeypatch.setattr(live, "read_cases", lambda *a, **k: pytest.fail("Opened heldout source"))
    monkeypatch.setattr(live, "model_snapshot_hashes", lambda: pytest.fail("Opened model"))
    monkeypatch.setattr(sys, "argv", ["live", "--mode", "heldout", "--domain", "klue",
                                      "--decision", str(decision_path), "--development",
                                      str(development), "--checkpoint", str(checkpoint),
                                      "--reserve", str(reserve), "--output",
                                      str(tmp_path / "result.json")])
    with pytest.raises(ValueError, match="both"):
        live.main()


def test_replay_requires_per_case_coordinates_and_stars_but_not_confidence():
    cases = [{"id": "a"}, {"id": "b"}]
    old = [{(0, 1)}, {(1, 2)}]
    findings = [[{"entity": "KR_NAME", "start": 0, "end": 1, "score": 1.}],
                [{"entity": "KR_NAME", "start": 1, "end": 2, "score": 1.}]]
    changed_scores = copy.deepcopy(findings)
    changed_scores[0][0]["score"] = .95
    result = live.replay_comparison(cases, old, ["*a", "a*"], findings,
                                    old, ["*a", "a*"], changed_scores)
    assert result["matched"] is True
    assert result["score_only_difference_cases"] == 1
    assert result["max_score_difference"] == pytest.approx(.05)
    result = live.replay_comparison(cases, old, ["*a", "a*"], findings,
                                    old[::-1], ["a*", "*a"], findings)
    assert result["matched"] is False
    assert result["span_disagreement_ids"] == ["a", "b"]
    assert result["mask_disagreement_ids"] == ["a", "b"]


def test_ancestry_overlap_checks_exact_and_both_sides_of_normalization(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps({"text": "Ａ씨"}) + "\n")
    original = hashlib.sha256("Ａ씨".encode()).hexdigest()
    write(checkpoint / "training-text-hashes.json", {
        "algorithm": "sha256_utf8_exact_text", "text_sha256": [original]})
    write(checkpoint / "manifest.json", {"source_sha256": {
        str(source.resolve()): live.digest(source)}})
    exact, normalized, hashes = live.load_ancestry(checkpoint)
    assert str(source.resolve()) in hashes
    cases = [{"id": "same", "text": "Ａ씨"}, {"id": "normalized", "text": "A씨"},
             {"id": "clean", "text": "김씨"}]
    result = live.ancestry_overlaps(cases, exact, normalized)
    assert result == {"exact_ids": ["same"], "normalized_ids": ["same", "normalized"]}


def test_ancestry_rejects_hashes_missing_from_pinned_training_sources(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    write(checkpoint / "training-text-hashes.json", {
        "algorithm": "sha256_utf8_exact_text", "text_sha256": ["a" * 64]})
    write(checkpoint / "manifest.json", {"source_sha256": {}})
    with pytest.raises(ValueError, match="ancestry"):
        live.load_ancestry(checkpoint)


def test_heldout_requires_both_successful_live_domains_and_same_artifact(tmp_path):
    _, _, _, _, _ = selected_fixture(tmp_path)
    identity = dict(decision_sha256="decision", development_report_sha256="development",
                    artifact_sha256={"weights": "fixed"}, selected_policy={"cutoff": .5},
                    reserve_sha256="reserve", device="cpu")
    reports = []
    for domain, names in live.DOMAIN_SOURCES.items():
        marker = tmp_path / (domain + "-fixed.txt")
        marker.write_text("fixed")
        consumed = write(tmp_path / (domain + "-consumed.json"), {"selected_ids": [domain]})
        reports.append(dict(scope="multisource_live_development", domain=domain, passed=True,
                            source_changed_during_run=False, cache_replay={"matched": True},
                            selected_ids=[domain], consumed_report=str(consumed),
                            frozen_sha256={str(marker.resolve()): live.digest(marker),
                                           str(consumed.resolve()): live.digest(consumed)},
                            sources={name: {"regression_gate": gate(), "passed": True,
                                            "aggregate_failures": []} for name in names},
                            **identity))
    live.validate_live_reports(reports, identity)
    with pytest.raises(ValueError, match="both"):
        live.validate_live_reports(reports[:1], identity)
    reports[1]["cache_replay"]["matched"] = False
    with pytest.raises(ValueError, match="replay|passed"):
        live.validate_live_reports(reports, identity)
    reports[1]["cache_replay"]["matched"] = True
    reports[1]["artifact_sha256"] = {"weights": "changed"}
    with pytest.raises(ValueError, match="artifact"):
        live.validate_live_reports(reports, identity)


def test_kdpii_reader_converts_only_predeclared_selected_ids(tmp_path, monkeypatch):
    source = tmp_path / "test.json"
    write(source, [{"sent_idx": "sealed", "PII_set": "MUST NOT BE CONVERTED"},
                   {"sent_idx": "selected"}])
    opened = []

    def convert(row):
        opened.append(row["sent_idx"])
        assert row["sent_idx"] == "selected"
        return {"id": row["sent_idx"], "text": "a", "expected": []}

    monkeypatch.setattr(live, "convert_record", convert)
    assert [r["id"] for r in live.read_cases("kdpii", "heldout", source, ["selected"])] == [
        "selected"]
    assert opened == ["selected"]


def test_cached_selected_policy_replays_real_stars_and_preserves_address_suppression():
    cases = [{"id": "one-nsmc", "text": "홍길동 왔다", "expected": [
        {"entity": "KR_NAME", "start": 0, "end": 3}]}]
    baseline = live._score(cases, [set()], [cases[0]["text"]])
    logits = torch.full((len(cases[0]["text"]), 5), -20.)
    logits[:, 0] = 20.
    for position, label in enumerate((1, 2, 3)):
        logits[position, 0], logits[position, label] = -20., 20.
    cache = {"selected_ids": ["one-nsmc"], "logits": [logits], "spans": [[]]}
    policy = dict(cutoff=.5, class_logit_correction=0., protect_anchors=False)
    old, candidate = live.cached_replay(cases, cache, baseline, policy)
    assert old[0] == [set()]
    assert candidate[0] == [{(0, 3)}]
    assert candidate[1] == ["*** 왔다"]
    cache["spans"] = [[dict(entity="KR_ADDRESS", start=0, end=3, score=.99)]]
    _, suppressed = live.cached_replay(cases, cache, baseline, policy)
    assert suppressed[0] == [set()]
    assert suppressed[1] == [cases[0]["text"]]


def test_selective_policy_uses_lazy_decoder_and_factory_with_fixed_thresholds(monkeypatch):
    policy = dict(algorithm="selective_whole_span", addition_threshold=.9,
                  removal_threshold=.95, class_logit_correction=0.)
    live.validate_policy(policy)
    captured = {}

    def selective(logits, addition_threshold, removal_threshold, anchors):
        captured.update(logits=logits, addition=addition_threshold,
                        removal=removal_threshold, anchors=anchors)
        return anchors

    def factory(config):
        return config
    monkeypatch.setitem(sys.modules, "name_selective_evidence",
                        types.SimpleNamespace(selective_spans=selective))
    monkeypatch.setitem(sys.modules, "name_selective_adapter",
                        types.SimpleNamespace(build_ner=factory))
    logits = torch.zeros(3, 5)
    assert live.decode_policy(logits, {(0, 2)}, policy) == [(0, 2, 1.)]
    assert captured["addition"] == .9 and captured["removal"] == .95
    assert captured["logits"].eq(0.).all()
    assert logits.eq(0.).all()
    assert live.resolve_factory(policy) == (factory, "name_selective_adapter:build_ner")
    with pytest.raises(ValueError):
        live.validate_policy(dict(policy, protect_anchors=True))
    with pytest.raises(ValueError, match="zero"):
        live.validate_policy(dict(policy, class_logit_correction=.4))


def test_selective_address_exception_applies_only_to_actually_retained_anchors(monkeypatch):
    cases = [{"id": "one-nsmc", "text": "홍길동 왔다", "expected": [
        {"entity": "KR_NAME", "start": 0, "end": 3}]}]
    baseline = live._score(cases, [{(0, 3)}], ["*** 왔다"])
    cache = {"selected_ids": ["one-nsmc"], "logits": [torch.zeros(6, 5)],
             "spans": [[dict(entity="KR_ADDRESS", start=0, end=3, score=.99)]]}
    policy = dict(algorithm="selective_whole_span", addition_threshold=.1,
                  removal_threshold=.95, class_logit_correction=0.)
    outside = [.9]

    def selective(logits, addition_threshold, removal_threshold, anchors):
        # Same coordinates may be a retained anchor or a newly re-proposed name.
        return [(0, 3, .99)]

    monkeypatch.setitem(sys.modules, "name_selective_evidence", types.SimpleNamespace(
        selective_spans=selective, all_o_posteriors=lambda *args: torch.tensor(outside)))
    _, kept = live.cached_replay(cases, cache, baseline, policy)
    assert kept[0] == [{(0, 3)}] and kept[1] == ["*** 왔다"]
    outside[0] = .99
    _, removed_then_reproposed = live.cached_replay(cases, cache, baseline, policy)
    assert removed_then_reproposed[0] == [set()]
    assert removed_then_reproposed[1] == [cases[0]["text"]]


def test_live_development_writes_manifest_before_read_and_reproduces_actual_masks(
    tmp_path, monkeypatch,
):
    checkpoint, development_path, decision_path, development, _ = selected_fixture(tmp_path)
    text = "홍길동 왔다"
    cases = [dict(id=f"fixture-{i}-" + ("nsmc" if i < 500 else "wikitree"), text=text,
                  expected=[dict(entity="KR_NAME", start=0, end=3)]) for i in range(1000)]
    source = tmp_path / "consumed.tsv"
    source.write_text("Only the injected consumed fixture reader may parse this file")
    exclusion = write(tmp_path / "exclusion.json", [])
    write(checkpoint / "data-manifest.json", {
        "input_sha256": {str(exclusion): live.digest(exclusion)}})
    consumed = write(tmp_path / "consumed-report.json", {
        "selected_ids": [row["id"] for row in cases],
        "frozen_sha256": {str(source): live.digest(source)},
        "baseline": {"metrics": live._score(cases, [set()] * 1000, [text] * 1000)},
    })
    reserve = write(tmp_path / "reserve.json", {
        "scope": "v2_evaluation_reserve_before_any_new_training", "test_labels_read": False,
        "klue": {"selected_ids": [f"sealed-klue-{i}" for i in range(1000)],
                 "consumed_ids": [row["id"] for row in cases]},
        "kdpii": {"selected_ids": [f"sealed-kdpii-{i}" for i in range(2000)],
                  "consumed_ids": []},
        "exclusion_hashes_path": str(exclusion), "exclusion_hashes_sha256": live.digest(exclusion),
    })
    for domain in ("nsmc", "wikitree"):
        rows = [row for row in cases if row["id"].endswith(domain)]
        development["baselines"][domain] = live._score(rows, [set()] * 500, [text] * 500)
        development["candidates"][0]["sources"][domain]["metrics"] = live._score(
            rows, [{(0, 3)}] * 500, ["*** 왔다"] * 500)
    development["frozen_sha256"].update({str(path.resolve()): live.digest(path)
                                        for path in [*checkpoint.iterdir(), consumed, source]})
    write(development_path, development)
    decision = select_candidates(development["baselines"], development["candidates"],
                                 required_sources=development["required_sources"])
    decision.update(development_report_sha256=live.digest(development_path),
                    checker_sha256=live.digest(live.ROOT / "scripts/name_multisource_selection.py"))
    write(decision_path, decision)
    logits = torch.full((len(text), 5), -20.)
    logits[:, 0] = 20.
    for position, label in enumerate((1, 2, 3)):
        logits[position, 0], logits[position, label] = -20., 20.
    cache = dict(selected_ids=[row["id"] for row in cases], logits=[logits] * 1000,
                 spans=[[] for _ in cases])
    output = tmp_path / "live.json"
    order = []

    def reader(domain, mode, path, identifiers):
        assert output.with_suffix(".manifest.json").exists()
        assert order == []
        assert identifiers == cache["selected_ids"]
        order.append("read_consumed")
        return cases

    class NER:
        device = "cpu"

        def __init__(self, detects):
            self.detects = detects

        def analyze(self, text):
            return [live.RecognizerResult("KR_NAME", 0, 3, .99)] if self.detects else []

    def baseline(**kwargs):
        assert kwargs["score_threshold"] == .9
        order.append("baseline_loaded")
        return NER(False)

    def factory(config):
        assert config["model_kind"] == "frozen" and config["cutoff"] == .5
        order.append("candidate_loaded")
        return NER(True)

    monkeypatch.setattr(live, "read_cases", reader)
    monkeypatch.setattr(live, "load_development_cache", lambda *args: cache)
    monkeypatch.setattr(live, "load_ancestry", lambda *args: (set(), set(), {}))
    monkeypatch.setattr(live, "model_snapshot_hashes", lambda: {})
    monkeypatch.setattr(live, "resolve_factory", lambda policy: (factory, "test:factory"))
    monkeypatch.setattr(live.KoreanNER, "from_pretrained", baseline)
    monkeypatch.setattr(sys, "argv", ["live", "--mode", "development", "--domain", "klue",
                                      "--decision", str(decision_path), "--development",
                                      str(development_path), "--checkpoint", str(checkpoint),
                                      "--reserve", str(reserve), "--source", str(source),
                                      "--consumed-report", str(consumed), "--output", str(output)])
    live.main()
    result = json.loads(output.read_text())
    assert result["passed"] is True
    assert result["cache_replay"]["matched"] is True
    assert result["cache_replay"]["candidate"]["score_only_difference_cases"] == 1000
    assert result["source_changed_during_run"] is False
    assert order == ["read_consumed", "baseline_loaded", "candidate_loaded"]
