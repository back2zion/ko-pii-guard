"""One frozen candidate must pass all fresh domains without hiding regressions."""

import copy
import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("presidio_analyzer")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import check_name_multisource_release as release
    import evaluate_name_multisource_live as live
    from name_multisource_selection import select_candidates
finally:
    sys.path[:] = previous_path


def write(path, value):
    path.write_text(json.dumps(value))
    return path


def score(sentences, improved=False):
    fp = 1 if improved else 2
    return dict(
        tp=8,
        fp=fp,
        fn=2,
        gold_names=10,
        fully_covered_names=8,
        unnecessary_masked_characters=2 if improved else 4,
        negative_false_positive_sentences=fp,
        f1=16 / (18 + fp),
        precision=8 / (8 + fp),
        recall=0.8,
        full_name_coverage=0.8,
        sentences=sentences,
        negative_sentences=sentences - 10,
    )


def gate():
    return dict(
        passed=True,
        lost_correct_spans=[],
        introduced_false_spans=[],
        newly_exposed_names=[],
        actual_stars_newly_exposed_names=[],
    )


def result(sentences, improved=False):
    return dict(
        baseline=score(sentences),
        metrics=score(sentences, improved),
        regression_gate=gate(),
        passed=True,
        aggregate_failures=[],
        bootstrap=dict(
            method="paired sentence percentile bootstrap",
            repetitions=500,
            seed=20261010,
            delta_95_intervals={
                key: [-0.1, 0.2] for key in ("f1", "precision", "recall", "full_name_coverage")
            },
        ),
    )


def fixture_inputs(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "head.safetensors").write_bytes(b"fixture only; never a loadable model")
    write(
        checkpoint / "config.json",
        dict(selected_epoch=1, threshold=0.9, model_id=live.MODEL_ID, revision=live.MODEL_REVISION),
    )
    write(
        checkpoint / "report.json",
        dict(source_changed=False, selected_epoch=1, threshold=0.9, history=[dict(epoch=1)]),
    )
    write(checkpoint / "manifest.json", dict(epochs=1))
    write(
        checkpoint / "training-text-hashes.json",
        dict(algorithm="sha256_utf8_exact_text", text_sha256=[]),
    )
    exclusions = write(tmp_path / "exclusions.json", {"fixture": True})
    write(
        checkpoint / "data-manifest.json",
        dict(input_sha256={str(exclusions): live.digest(exclusions)}),
    )
    consumed, selected = {}, {}
    for domain, count in (("klue", 1000), ("kdpii", 500)):
        ids = [
            f"old-{i}-" + ("nsmc" if i % 2 else "wikitree") if domain == "klue" else f"old-kd-{i}"
            for i in range(count)
        ]
        consumed[domain] = write(tmp_path / f"consumed-{domain}.json", {"selected_ids": ids})
        selected[domain] = [
            f"fresh-{i}-" + ("nsmc" if i % 2 else "wikitree")
            if domain == "klue"
            else f"fresh-kd-{i}"
            for i in range(1000 if domain == "klue" else 2000)
        ]
    source = tmp_path / "source.bin"
    source.write_bytes(b"fixture provenance marker, no evaluation text")
    reserve_data = dict(
        scope="v2_evaluation_reserve_before_any_new_training",
        test_labels_read=False,
        source_sha256={str(source): live.digest(source)},
        exclusion_hashes_path=str(exclusions),
        exclusion_hashes_sha256=live.digest(exclusions),
    )
    for domain in selected:
        reserve_data[domain] = dict(
            selected_ids=selected[domain],
            consumed_ids=live.read_json(consumed[domain])["selected_ids"],
        )
    reserve = write(tmp_path / "reserve.json", reserve_data)
    policy = dict(
        algorithm="selective_whole_span",
        addition_threshold=1.0,
        removal_threshold=0.9999,
        class_logit_correction=0.0,
    )
    artifact = {str(p.resolve()): live.digest(p) for p in checkpoint.iterdir()}
    pinned = dict(artifact)
    runtime_scripts = (
        "evaluate_name_multisource_live.py",
        "name_selective_adapter.py",
        "name_selective_evidence.py",
        "name_multisource_adapter.py",
        "name_multisource_selection.py",
        "name_span_evidence.py",
        "name_adapted_adapter.py",
        "name_generalization_model.py",
        "name_bioes_ablation.py",
        "evaluate_name_generalization.py",
    )
    for path in [
        *(live.ROOT / "scripts" / name for name in runtime_scripts),
        *(live.ROOT / "src/ko_pii_guard").glob("*.py"),
        source,
        exclusions,
        *consumed.values(),
    ]:
        pinned[str(path.resolve())] = live.digest(path)
    snapshot = (
        tmp_path
        / ("models--" + live.MODEL_ID.replace("/", "--"))
        / "snapshots"
        / live.MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"fixture encoder, never loaded")
    write(snapshot / "config.json", {"fixture": True})
    for path in snapshot.iterdir():
        pinned[str(path)] = live.digest(path)
    candidate_sources = {
        name: dict(metrics=score(500, name == "wikitree"), regression_gate=gate())
        for name in live.REQUIRED_SOURCES
    }
    development_data = dict(
        scope="multisource_development_only",
        baseline_replay_verified=True,
        source_changed_during_run=False,
        required_sources=live.REQUIRED_SOURCES,
        baselines={name: score(500) for name in live.REQUIRED_SOURCES},
        candidates=[dict(id="fixed", policy=policy, sources=candidate_sources)],
        frozen_sha256=pinned,
    )
    development = write(tmp_path / "development.json", development_data)
    selected_decision = select_candidates(
        development_data["baselines"],
        development_data["candidates"],
        required_sources=live.REQUIRED_SOURCES,
    )
    selected_decision.update(
        development_report_sha256=live.digest(development),
        checker_sha256=live.digest(live.ROOT / "scripts/name_multisource_selection.py"),
    )
    decision = write(tmp_path / "decision.json", selected_decision)
    runtime = dict(
        python="3.10.fixture",
        torch="fixture",
        packages={name: "fixture" for name in ("transformers", "presidio-analyzer", "safetensors")},
    )
    identity = dict(
        decision_sha256=live.digest(decision),
        development_report_sha256=live.digest(development),
        artifact_sha256=artifact,
        selected_policy=policy,
        reserve_sha256=live.digest(reserve),
        device="cpu",
        runtime_signature=runtime,
    )
    paths = dict(
        decision=decision,
        development=development,
        checkpoint=checkpoint,
        reserve=reserve,
        live_development=[],
        heldout=[],
    )
    for path in (decision, development, reserve):
        pinned[str(path.resolve())] = live.digest(path)
    for mode in ("development", "heldout"):
        for domain in ("klue", "kdpii"):
            ids = (
                live.read_json(consumed[domain])["selected_ids"]
                if mode == "development"
                else selected[domain]
            )
            source_rows = {
                name: result(
                    500 if mode == "development" or domain == "klue" else 2000, name == "wikitree"
                )
                for name in live.DOMAIN_SOURCES[domain]
            }
            row = dict(
                scope="multisource_live_" + mode,
                domain=domain,
                passed=True,
                selected_id="fixed",
                selected_ids=ids,
                consumed_report=str(consumed[domain]),
                source_changed_during_run=False,
                changed_inputs=[],
                sources=source_rows,
                ancestry_overlap=dict(exact_ids=[], normalized_ids=[]),
                **identity,
                runtime={**runtime, "devices": dict(baseline="cpu", candidate="cpu")},
                candidate_factory="name_selective_adapter:build_ner",
                model_kind="frozen",
                baseline_model=dict(
                    model_id=live.MODEL_ID, revision=live.MODEL_REVISION, threshold=0.9
                ),
                guard_score_threshold=0.0,
                runtime_promotion=False,
                resampling=False,
                frozen_sha256=dict(pinned),
            )
            if mode == "development":
                comparison = dict(matched=True, span_disagreement_ids=[], mask_disagreement_ids=[])
                row["cache_replay"] = dict(
                    matched=True,
                    metric_differences=[],
                    baseline=copy.deepcopy(comparison),
                    candidate=copy.deepcopy(comparison),
                )
            else:
                row["frozen_sha256"].update(
                    {str(p): live.digest(p) for p in paths["live_development"]}
                )
            path = write(tmp_path / f"{mode}-{domain}.json", row)
            paths["live_development" if mode == "development" else "heldout"].append(path)
    return paths


def mutate(paths, group, index, function):
    path = paths[group][index]
    report = live.read_json(path)
    function(report)
    write(path, report)


def checks(decision):
    return {row["check"] for row in decision["reasons"]}


def test_fresh_improvement_passes_even_when_ci_crosses_zero_without_promoting(
    tmp_path, monkeypatch
):
    paths = fixture_inputs(tmp_path)
    for name in ("read_cases", "load_ancestry", "model_snapshot_hashes", "resolve_factory"):
        monkeypatch.setattr(live, name, lambda *a, **k: pytest.fail("Opened raw data/model"))
    decision = release.evaluate_release(**paths)
    assert decision["status"] == "quality_gate_passed", decision["reasons"]
    assert decision["quality_gate_passed"] is True
    assert decision["automatic_runtime_promotion"] is False
    assert decision["historical_contracts_checked"] is False
    assert decision["confidence_intervals"]["wikitree"]["delta_95_intervals"]["f1"] == [-0.1, 0.2]


def test_equal_fresh_metrics_cannot_reuse_development_improvement(tmp_path):
    paths = fixture_inputs(tmp_path)
    mutate(paths, "heldout", 0, lambda r: r["sources"]["wikitree"].update(metrics=score(500)))
    decision = release.evaluate_release(**paths)
    assert not decision["quality_gate_passed"]
    assert "strict_improvement" in checks(decision)


def test_concrete_stars_regression_fails_despite_passed_flags_and_aggregate_gain(tmp_path):
    paths = fixture_inputs(tmp_path)
    mutate(
        paths,
        "heldout",
        0,
        lambda r: r["sources"]["wikitree"]["regression_gate"].update(
            actual_stars_newly_exposed_names=[["fixture", 0, 1]]
        ),
    )
    decision = release.evaluate_release(**paths)
    assert not decision["quality_gate_passed"]
    assert "strict_regression_gate" in checks(decision)


def test_one_kdpii_policy_degradation_cannot_be_pooled_away(tmp_path):
    paths = fixture_inputs(tmp_path)
    mutate(
        paths,
        "heldout",
        1,
        lambda r: r["sources"]["kdpii/PS_NAME+PS_NICKNAME"]["metrics"].update(
            unnecessary_masked_characters=5
        ),
    )
    decision = release.evaluate_release(**paths)
    assert not decision["quality_gate_passed"]
    assert "aggregate_constraints" in checks(decision)


@pytest.mark.parametrize(
    "change,expected",
    [
        (lambda r: r.update(scope="multisource_live_development"), "completed_report"),
        (lambda r: r.update(passed=False), "completed_report"),
        (lambda r: r.update(source_changed_during_run=True), "completed_report"),
        (lambda r: r.update(selected_id="other"), "same_candidate"),
        (lambda r: r["selected_policy"].update(removal_threshold=0.9), "same_candidate"),
        (lambda r: r["artifact_sha256"].clear(), "same_candidate"),
        (lambda r: r["runtime_signature"].update(torch="different"), "same_candidate"),
        (lambda r: r["runtime"]["devices"].update(candidate="cuda:0"), "runtime"),
        (lambda r: r.update(candidate_factory="different:factory"), "same_candidate"),
        (lambda r: r.update(resampling=True), "completed_report"),
        (lambda r: r["ancestry_overlap"].update(normalized_ids=["duplicate"]), "training_overlap"),
        (lambda r: r["ancestry_overlap"].pop("exact_ids"), "training_overlap"),
        (lambda r: r["selected_ids"].reverse(), "fixed_population"),
        (lambda r: r["sources"].pop("wikitree"), "source_complete"),
        (lambda r: r.update(sources=None), "source_complete"),
        (lambda r: r["sources"].update(wikitree=None), "source_complete"),
        (lambda r: r["sources"]["wikitree"]["metrics"].update(f1=float("nan")), "valid_metrics"),
        (
            lambda r: r["sources"]["wikitree"]["metrics"].update(full_name_coverage=0.9),
            "valid_metrics",
        ),
        (lambda r: r["sources"]["wikitree"]["metrics"].update(sentences=499), "fixed_population"),
    ],
)
def test_fresh_completion_identity_population_and_metric_checks(tmp_path, change, expected):
    paths = fixture_inputs(tmp_path)
    mutate(paths, "heldout", 0, change)
    decision = release.evaluate_release(**paths)
    assert not decision["quality_gate_passed"]
    assert expected in checks(decision)


def test_missing_live_report_and_unpinned_live_predecessor_are_rejected(tmp_path):
    paths = fixture_inputs(tmp_path)
    incomplete = dict(paths, live_development=paths["live_development"][:1])
    assert not release.evaluate_release(**incomplete)["quality_gate_passed"]
    mutate(paths, "heldout", 0, lambda r: r["frozen_sha256"].pop(str(paths["live_development"][0])))
    decision = release.evaluate_release(**paths)
    assert "frozen_inputs" in checks(decision)


def test_passed_live_flag_cannot_hide_replay_disagreement(tmp_path):
    paths = fixture_inputs(tmp_path)
    mutate(
        paths,
        "live_development",
        0,
        lambda r: r["cache_replay"]["candidate"].update(mask_disagreement_ids=["fixture"]),
    )
    decision = release.evaluate_release(**paths)
    assert not decision["quality_gate_passed"]
    assert "live_replay" in checks(decision)


def test_changed_checkpoint_or_reserve_source_is_rejected(tmp_path):
    paths = fixture_inputs(tmp_path)
    (tmp_path / "source.bin").write_bytes(b"modified")
    assert not release.evaluate_release(**paths)["quality_gate_passed"]


def test_missing_encoder_snapshot_hashes_cannot_claim_same_runtime(tmp_path):
    paths = fixture_inputs(tmp_path)

    def remove_encoder_pins(report):
        report["frozen_sha256"] = {
            p: sha for p, sha in report["frozen_sha256"].items() if "/snapshots/" not in p
        }

    mutate(paths, "heldout", 0, remove_encoder_pins)
    decision = release.evaluate_release(**paths)
    assert not decision["quality_gate_passed"]
    assert "encoder_snapshot" in checks(decision)


def test_no_candidate_selection_is_rejected_without_reading_heldout(tmp_path, monkeypatch):
    paths = fixture_inputs(tmp_path)
    decision = live.read_json(paths["decision"])
    write(paths["decision"], dict(decision, status="no_candidate"))
    paths["heldout"] = [tmp_path / "must-not-open-one", tmp_path / "must-not-open-two"]
    outcome = release.evaluate_release(**paths)
    assert not outcome["quality_gate_passed"]
    assert checks(outcome) == {"selected_decision"}


def test_cli_records_quality_failure_and_refuses_to_overwrite(tmp_path, monkeypatch):
    paths = fixture_inputs(tmp_path)
    mutate(paths, "heldout", 0, lambda r: r["sources"]["wikitree"].update(metrics=score(500)))
    output = tmp_path / "release.json"
    args = ["checker"]
    for key in ("decision", "development", "checkpoint", "reserve"):
        args += ["--" + key, str(paths[key])]
    for key in ("live_development", "heldout"):
        args += ["--" + key.replace("_", "-"), *map(str, paths[key])]
    args += ["--output", str(output), "--check"]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as exit_info:
        release.main()
    assert exit_info.value.code == 1
    outcome = live.read_json(output)
    assert outcome["status"] == "quality_gate_failed"
    assert outcome["checker_sha256"] == live.digest(release.__file__)
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        release.main()
    assert output.read_bytes() == original


def amended_fixture(tmp_path, monkeypatch):
    paths = fixture_inputs(tmp_path)
    reserve = live.read_json(paths["reserve"])
    amendment = dict(
        scope="multisource_reserve_text_amendment",
        original_reserve_sha256=live.digest(paths["reserve"]),
        created_at_utc="2026-10-10T00:00:00+00:00",
        domains={},
    )
    for domain in ("klue", "kdpii"):
        original = reserve[domain]["selected_ids"]
        excluded = original[-2:] if domain == "kdpii" else []
        selected = original[:-2] if excluded else original[:]
        amendment["domains"][domain] = dict(
            original_selected_ids=original,
            selected_ids=selected,
            excluded_ids=excluded,
            original_count=len(original),
            selected_count=len(selected),
            excluded_count=len(excluded),
        )
    amendment_path = write(tmp_path / "amendment.json", amendment)
    helper_path = tmp_path / "fixture-text-audit.py"
    helper_path.write_text("# Fixture delegate; no source text is opened\n")
    calls = []

    def validate(amendment_arg, reserve_arg):
        calls.append((Path(amendment_arg), Path(reserve_arg)))
        return live.read_json(amendment_arg)

    monkeypatch.setitem(
        sys.modules,
        "name_reserve_text_audit",
        types.SimpleNamespace(__file__=str(helper_path), validate_amendment=validate),
    )
    # The driver source is pinned in real reports. Isolate this fixture from its
    # implementation; audit delegate correctness has separate text-audit tests.
    driver_path = tmp_path / "fixture-fresh-driver.py"
    driver_path.write_text("# Fixture completed-report provenance\n")
    monkeypatch.setattr(release, "FRESH_DRIVER", driver_path)
    for index, domain in enumerate(("klue", "kdpii")):
        row = live.read_json(paths["heldout"][index])
        audited = amendment["domains"][domain]
        row.update(
            selected_ids=audited["selected_ids"],
            original_selected_ids=audited["original_selected_ids"],
            excluded_ids=audited["excluded_ids"],
            amendment_sha256=live.digest(amendment_path),
            amendment_path=str(amendment_path.resolve()),
            amendment_created_at_utc=amendment["created_at_utc"],
            original_selected_count=audited["original_count"],
            selected_count=audited["selected_count"],
            excluded_count=audited["excluded_count"],
        )
        row["frozen_sha256"].update(
            {str(p.resolve()): live.digest(p) for p in (amendment_path, helper_path, driver_path)}
        )
        if domain == "kdpii":
            for name in row["sources"]:
                row["sources"][name] = result(len(audited["selected_ids"]))
        manifest_path = paths["heldout"][index].with_suffix(".manifest.json")
        manifest = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "sources",
                "passed",
                "source_changed_during_run",
                "changed_inputs",
                "ancestry_overlap",
            }
        }
        write(manifest_path, manifest)
        row.update(
            pre_inference_manifest_path=str(manifest_path),
            pre_inference_manifest_sha256=live.digest(manifest_path),
        )
        row["frozen_sha256"][str(manifest_path)] = live.digest(manifest_path)
        write(paths["heldout"][index], row)
    paths["amendment"] = amendment_path
    return paths, calls


def test_recomputed_text_amendment_allows_only_fixed_retained_ids(tmp_path, monkeypatch):
    paths, calls = amended_fixture(tmp_path, monkeypatch)
    decision = release.evaluate_release(**paths)
    assert decision["quality_gate_passed"], decision["reasons"]
    assert calls == [(paths["amendment"], paths["reserve"])]
    assert decision["amendment_sha256"] == live.digest(paths["amendment"])
    assert decision["evaluated_selected_counts"] == {"klue": 1000, "kdpii": 1998}
    without_authorization = dict(paths)
    without_authorization.pop("amendment")
    assert not release.evaluate_release(**without_authorization)["quality_gate_passed"]


@pytest.mark.parametrize(
    "change,expected",
    [
        (lambda r: r["selected_ids"].append("replacement"), "fixed_population"),
        (lambda r: r.update(excluded_ids=[]), "amendment_provenance"),
        (lambda r: r.update(amendment_created_at_utc="later"), "amendment_provenance"),
        (lambda r: r.update(original_selected_count=1998), "amendment_provenance"),
        (lambda r: r.update(amendment_sha256="changed"), "amendment_provenance"),
        (lambda r: r["frozen_sha256"].pop(r["amendment_path"]), "frozen_inputs"),
        (lambda r: r.pop("pre_inference_manifest_sha256"), "pre_inference_amendment"),
    ],
)
def test_amendment_cannot_hide_replacement_or_missing_pre_inference_record(
    tmp_path, monkeypatch, change, expected
):
    paths, _ = amended_fixture(tmp_path, monkeypatch)
    mutate(paths, "heldout", 1, change)
    decision = release.evaluate_release(**paths)
    assert not decision["quality_gate_passed"]
    assert expected in checks(decision)


def test_recomputed_amendment_failure_blocks_before_heldout_reports(tmp_path, monkeypatch):
    paths, _ = amended_fixture(tmp_path, monkeypatch)

    def invalid(*args):
        raise ValueError("Text-only audit no longer reproduces the frozen exclusions")

    monkeypatch.setattr(sys.modules["name_reserve_text_audit"], "validate_amendment", invalid)
    paths["heldout"] = [tmp_path / "must-not-open-one", tmp_path / "must-not-open-two"]
    decision = release.evaluate_release(**paths)
    assert not decision["quality_gate_passed"]
    assert "amendment_provenance" in checks(decision)


def test_pre_inference_manifest_must_pin_model_and_amendment_before_results(tmp_path, monkeypatch):
    paths, _ = amended_fixture(tmp_path, monkeypatch)
    report = live.read_json(paths["heldout"][0])
    manifest_path = Path(report["pre_inference_manifest_path"])
    manifest = live.read_json(manifest_path)
    manifest["frozen_sha256"].pop(str(paths["checkpoint"] / "head.safetensors"))
    write(manifest_path, manifest)
    # Even a newly updated manifest hash cannot hide missing pre-inference pins.
    report["pre_inference_manifest_sha256"] = live.digest(manifest_path)
    report["frozen_sha256"][str(manifest_path)] = live.digest(manifest_path)
    write(paths["heldout"][0], report)
    decision = release.evaluate_release(**paths)
    assert not decision["quality_gate_passed"]
    assert "pre_inference_amendment" in checks(decision)


def lossless_fixture(tmp_path, monkeypatch):
    paths, calls = amended_fixture(tmp_path, monkeypatch)
    original_manifest = paths["heldout"][1].with_suffix(".manifest.json")
    failed_manifest = tmp_path / "original-kdpii.manifest.json"
    failed_manifest.write_bytes(original_manifest.read_bytes())
    source_paths = {}
    for constant, field in (
        ("LOSSLESS_FRESH_DRIVER", "driver"),
        ("LOSSLESS_HELPER", "helper"),
        ("REFERENCE_CONVERTER", "reference_converter"),
    ):
        path = tmp_path / (field + ".py")
        path.write_text("# Isolated provenance fixture; no annotation/model access\n")
        monkeypatch.setattr(release, constant, path, raising=False)
        source_paths[field] = path
    source_paths.update(original_driver=release.FRESH_DRIVER,
                        original_failure_manifest=failed_manifest)
    row = live.read_json(paths["heldout"][1])
    compatibility = dict(
        version="identical_annotation_duplicates_v1",
        duplicate_records=[dict(id=row["selected_ids"][0], duplicates=[
            dict(label="CV_POSITION", start=37, end=39, indices=[0, 3])
        ])],
        selected_count=len(row["selected_ids"]), dropped_records=0, gold_policy_changed=False,
    )
    for prefix, path in source_paths.items():
        compatibility[prefix + "_path"] = str(path.resolve())
        compatibility[prefix + "_sha256"] = live.digest(path)
        row["frozen_sha256"][str(path.resolve())] = live.digest(path)
    row["parser_compatibility"] = compatibility
    # A new pre-inference manifest follows the failed original manifest, retaining
    # its identity and all source instances. Neither contains model predictions.
    manifest = live.read_json(original_manifest)
    manifest["parser_compatibility"] = copy.deepcopy(compatibility)
    manifest["frozen_sha256"].update({str(path.resolve()): live.digest(path)
                                     for path in source_paths.values()})
    write(original_manifest, manifest)
    row["pre_inference_manifest_sha256"] = live.digest(original_manifest)
    row["frozen_sha256"][str(original_manifest)] = live.digest(original_manifest)
    write(paths["heldout"][1], row)
    paths["kdpii_lossless_parser"] = True
    return paths, calls


def test_explicit_lossless_parser_preserves_population_and_requires_pre_inference_provenance(
    tmp_path, monkeypatch
):
    paths, _ = lossless_fixture(tmp_path, monkeypatch)
    decision = release.evaluate_release(**paths)
    assert decision["quality_gate_passed"], decision["reasons"]
    assert decision["evaluated_selected_counts"]["kdpii"] == 1998
    assert decision["automatic_runtime_promotion"] is False
    paths["kdpii_lossless_parser"] = False
    decision = release.evaluate_release(**paths)
    assert not decision["quality_gate_passed"]
    assert "parser_compatibility" in checks(decision)


@pytest.mark.parametrize("change", [
    lambda r: r.pop("parser_compatibility"),
    lambda r: r["parser_compatibility"].update(version="unreviewed_parser"),
    lambda r: r["parser_compatibility"].update(dropped_records=1),
    lambda r: r["parser_compatibility"].update(gold_policy_changed=True),
    lambda r: r["parser_compatibility"].update(selected_count=1997),
    lambda r: r["parser_compatibility"].update(helper_path="unreviewed.py"),
    lambda r: r["parser_compatibility"].update(driver_sha256="changed"),
    lambda r: r["parser_compatibility"]["duplicate_records"][0].update(id="replacement"),
    lambda r: r["parser_compatibility"]["duplicate_records"][0]["duplicates"][0].update(
        indices=[0, 0]),
    lambda r: r["frozen_sha256"].pop(r["parser_compatibility"]["helper_path"]),
])
def test_lossless_parser_does_not_waive_policy_population_or_source_integrity(
    tmp_path, monkeypatch, change
):
    paths, _ = lossless_fixture(tmp_path, monkeypatch)
    mutate(paths, "heldout", 1, change)
    decision = release.evaluate_release(**paths)
    assert not decision["quality_gate_passed"]
    assert checks(decision) & {"parser_compatibility", "frozen_inputs", "pre_inference_amendment"}


def test_lossless_parser_cannot_be_introduced_after_inference(tmp_path, monkeypatch):
    paths, _ = lossless_fixture(tmp_path, monkeypatch)
    report = live.read_json(paths["heldout"][1])
    manifest_path = Path(report["pre_inference_manifest_path"])
    manifest = live.read_json(manifest_path)
    manifest.pop("parser_compatibility")
    write(manifest_path, manifest)
    report["pre_inference_manifest_sha256"] = live.digest(manifest_path)
    report["frozen_sha256"][str(manifest_path)] = live.digest(manifest_path)
    write(paths["heldout"][1], report)
    decision = release.evaluate_release(**paths)
    assert "pre_inference_amendment" in checks(decision)


def test_lossless_parser_previous_failed_manifest_must_pin_the_same_candidate(
    tmp_path, monkeypatch
):
    paths, _ = lossless_fixture(tmp_path, monkeypatch)
    report = live.read_json(paths["heldout"][1])
    compatibility = report["parser_compatibility"]
    failed = Path(compatibility["original_failure_manifest_path"])
    original = live.read_json(failed)
    original["selected_policy"]["removal_threshold"] = 0.9
    write(failed, original)
    # Even refreshing the recorded digest cannot authorize a different candidate.
    compatibility["original_failure_manifest_sha256"] = live.digest(failed)
    report["frozen_sha256"][str(failed)] = live.digest(failed)
    write(paths["heldout"][1], report)
    decision = release.evaluate_release(**paths)
    assert "parser_compatibility" in checks(decision)
