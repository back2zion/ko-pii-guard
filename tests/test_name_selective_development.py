"""Selective development policies preserve baseline replay and frozen provenance."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
RecognizerResult = pytest.importorskip("presidio_analyzer").RecognizerResult
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from evaluate_name_generalization import actual_mask_gate, evaluate_guard
    from evaluate_name_selective_development import (
        assert_identity_policy,
        domains_for,
        policy_grid,
        rows_from_selected,
        validate_cache_collection,
    )
    from evaluate_name_span_evidence_development import compact, digest, guard_for
    from name_multisource_selection import select_candidates
    from name_selective_evidence import all_o_posteriors, select_selective_evidence, selective_spans
    from name_span_evidence import span_posteriors
finally:
    sys.path[:] = previous_path


def case(identifier="one", text="홍길동 왔다"):
    return dict(id=identifier, text=text,
                expected=[dict(entity="KR_NAME", start=0, end=3)],
                alternate_expected=[dict(entity="KR_NAME", start=0, end=3)])


def test_predeclared_grid_contains_twenty_unique_zero_correction_policies():
    policies = policy_grid()
    assert len(policies) == 20
    assert {p["addition_threshold"] for p in policies} == {.99, .999, .9999, 1.}
    assert {p["removal_threshold"] for p in policies} == {.9, .99, .999, .9999, 1.}
    assert all(p["class_logit_correction"] == 0 for p in policies)
    assert all(p["algorithm"] == "selective_whole_span" for p in policies)
    assert len({tuple(p.items()) for p in policies}) == 20


def test_preserved_anchor_is_not_revetoed_by_candidate_address():
    addresses = [[RecognizerResult("KR_ADDRESS", 0, 8, .99)]]
    selected = [[(0, 3, 1.), (4, 6, .9999), (9, 11, .9999)]]
    rows = rows_from_selected(selected, [{(0, 3)}], addresses)
    names = [(r.start, r.end) for r in rows[0] if r.entity_type == "KR_NAME"]
    assert names == [(0, 3), (9, 11)]
    removed = rows_from_selected(selected, [set()], addresses)
    assert [(r.start, r.end) for r in removed[0] if r.entity_type == "KR_NAME"] == [(9, 11)]


def test_identity_policy_must_preserve_per_case_spans_and_actual_masks():
    cases = [case()]
    baseline_rows = [[RecognizerResult("KR_NAME", 0, 3, 1.)]]
    score, predictions, masks = evaluate_guard(cases, guard_for(cases, baseline_rows))
    assert masks == ["*** 왔다"]
    assert_identity_policy(score, predictions, masks, compact(score), predictions, masks, "test")
    with pytest.raises(ValueError, match="mask"):
        assert_identity_policy(score, predictions, ["홍길동 왔다"], compact(score),
                               predictions, masks, "test")
    with pytest.raises(ValueError, match="predictions"):
        assert_identity_policy(score, [set()], masks, compact(score), predictions, masks, "test")


def test_identity_candidate_is_rejected_by_existing_strict_improvement_gate():
    cases = [case()]
    score, predictions, masks = evaluate_guard(cases, guard_for(
        cases, [[RecognizerResult("KR_NAME", 0, 3, 1.)]]))
    gate = actual_mask_gate(cases, predictions, predictions, masks, masks)
    identity = dict(id="identity", policy=policy_grid()[-1],
                    sources={"test": dict(metrics=compact(score), regression_gate=gate)})
    decision = select_candidates({"test": compact(score)}, [identity], required_sources=["test"])
    assert decision["status"] == "no_candidate"
    assert [r["check"] for r in decision["rejections"]["identity"]] == ["strict_improvement"]


def test_domain_mapping_has_exact_four_views_and_preserves_original_cases():
    klue = [case("first-nsmc"), case("second-wikitree")]
    kdpii = [dict(case("third"), alternate_expected=[])]
    domains = domains_for(klue, kdpii)
    assert list(domains) == ["nsmc", "wikitree", "kdpii/PS_NAME", "kdpii/PS_NAME+PS_NICKNAME"]
    assert domains["kdpii/PS_NAME"][0] == [2]
    assert domains["kdpii/PS_NAME+PS_NICKNAME"][1][0]["expected"] == []
    assert kdpii[0]["expected"]
    with pytest.raises(ValueError, match="source"):
        domains_for([case("unknown")], kdpii)


def test_both_cache_bytes_and_embedded_span_rows_require_completed_provenance(tmp_path):
    cache_path, spans_path = tmp_path / "cache.pt", tmp_path / "spans.jsonl"
    cache_path.write_text("pinned fixture")
    spans_path.write_text('{"id": "one", "spans": []}\n')
    proof = dict(scope="consumed_development_cache", source_changed_during_run=False,
                 source_sha256={str(p.resolve()): digest(p) for p in (cache_path, spans_path)})
    cache = dict(scope="consumed_development_cache", domain="klue", selected_ids=["one"],
                 source_changed_during_run=False, spans=[[]])
    assert validate_cache_collection(cache, proof, cache_path, spans_path, "klue") == [[]]
    with pytest.raises(ValueError, match="spans"):
        validate_cache_collection(dict(cache, spans=[[dict(entity="KR_NAME", start=0, end=1)]]),
                                  proof, cache_path, spans_path, "klue")
    with pytest.raises(ValueError, match="domain"):
        validate_cache_collection(cache, proof, cache_path, spans_path, "kdpii")
    with pytest.raises(ValueError, match="completed"):
        validate_cache_collection(dict(cache, source_changed_during_run=None), proof,
                                  cache_path, spans_path, "klue")
    cache_path.write_text("tampered fixture")
    with pytest.raises(ValueError, match="provenance"):
        validate_cache_collection(cache, proof, cache_path, spans_path, "klue")


def test_frozen_math_api_connects_to_public_masks_and_full_regression_gate():
    example = dict(case(text="홍길동과 김철수"),
                   expected=[dict(entity="KR_NAME", start=5, end=8)])
    logits = torch.full((8, 5), -30.)
    logits[:, 0] = 30.
    for index, tag in ((5, 1), (6, 2), (7, 3)):
        logits[index] = -30.
        logits[index, tag] = 30.
    anchors = [(0, 3, 1.)]
    posterior = span_posteriors(logits)
    outside = all_o_posteriors(logits, [(0, 3)])
    selected = select_selective_evidence(posterior, outside, .99, .99, anchors)
    assert selected == selective_spans(logits, .99, .99, anchors)
    assert [(start, end) for start, end, _ in selected] == [(5, 8)]
    baseline_score, baseline_predictions, baseline_masks = evaluate_guard(
        [example], guard_for([example], [[RecognizerResult("KR_NAME", 0, 3, 1.)]]))
    rows = rows_from_selected([selected], [set()], [[]])
    score, predictions, masks = evaluate_guard([example], guard_for([example], rows))
    assert baseline_masks == ["***과 김철수"]
    assert masks == ["홍길동과 ***"]
    assert score["tp"] == 1 and score["fp"] == score["fn"] == 0
    assert actual_mask_gate([example], baseline_predictions, predictions,
                            baseline_masks, masks)["passed"] is True
    identity = select_selective_evidence(posterior, outside, 1., 1., anchors)
    address_rows = [[RecognizerResult("KR_ADDRESS", 0, 8, .99)]]
    rows = rows_from_selected([identity], [{(0, 3)}], address_rows)
    score, predictions, masks = evaluate_guard([example], guard_for([example], rows))
    assert_identity_policy(score, predictions, masks, compact(baseline_score),
                           baseline_predictions, baseline_masks, "connected_identity")
