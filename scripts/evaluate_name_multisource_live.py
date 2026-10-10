"""Live public-API validation of a selected research policy, with sealed reserves.

Development must reproduce cached coordinates and actual stars on every case.
Held-out labels are opened only after both live development domains pass with
the same immutable policy, model, reserve, and runtime inputs. No resampling,
policy search, or automatic runtime promotion occurs in this command.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch
from diagnose_name_generalization import reconstruct
from evaluate_name_generalization import (
    _score,
    actual_mask_gate,
    evaluate_guard,
    paired_intervals,
    read_selected_cases,
    validate_devices,
)
from evaluate_name_kdpii import policy_cases
from evaluate_name_kdpii import validate_cases as validate_kdpii_cases
from evaluate_name_span_evidence_development import compact, guard_for, validate_cache_cases
from fetch_kdpii_evaluation import convert_record
from name_multisource_adapter import _validate_policy
from name_multisource_selection import REGRESSIONS, select_candidates
from name_span_evidence import evidence_spans
from presidio_analyzer import RecognizerResult

from ko_pii_guard import KoreanPIIGuard
from ko_pii_guard.ner import MODEL_ID, MODEL_REVISION, KoreanNER
from ko_pii_guard.normalization import normalize_text

ROOT = Path(__file__).resolve().parents[1]
DOMAIN_SOURCES = {"klue": ("nsmc", "wikitree"),
                  "kdpii": ("kdpii/PS_NAME", "kdpii/PS_NAME+PS_NICKNAME")}
REQUIRED_SOURCES = (*DOMAIN_SOURCES["klue"], *DOMAIN_SOURCES["kdpii"])
CONSUMED_REPORTS = {
    "klue": ROOT / "benchmarks/results/name-generalization-final-v1-nonoverlap-development.json",
    "kdpii": ROOT / "benchmarks/results/name-generalization-final-v1-nonoverlap-kdpii.json",
}
TRAINING_FILES = {"train.jsonl", "validation.jsonl", "klue-validation.jsonl",
                  "kdpii-validation.jsonl", "synthetic-validation.jsonl"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_hashes(hashes):
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Require nonempty frozen hashes")
    for path, expected in hashes.items():
        if not Path(path).is_file() or digest(path) != expected:
            raise ValueError(f"Frozen input changed or is missing: {path}")


def validate_policy(policy):
    if policy.get("algorithm") == "selective_whole_span":
        if set(policy) != {"algorithm", "addition_threshold", "removal_threshold",
                           "class_logit_correction"}:
            raise ValueError("Require the complete selective whole-span policy")
        for key in ("addition_threshold", "removal_threshold"):
            _validate_policy(policy[key], policy["class_logit_correction"], True)
        if policy["class_logit_correction"] != 0.:
            raise ValueError("Selective unweighted-head policy requires zero logit correction")
    else:
        if set(policy) != {"cutoff", "class_logit_correction", "protect_anchors"}:
            raise ValueError("Require the explicit complete whole-span policy")
        _validate_policy(policy["cutoff"], policy["class_logit_correction"],
                         policy["protect_anchors"])


def resolve_factory(policy):
    name = ("name_selective_adapter" if policy.get("algorithm") == "selective_whole_span"
            else "name_multisource_adapter")
    return importlib.import_module(name).build_ner, name + ":build_ner"


def decode_policy(logits, anchors, policy):
    corrected = logits.clone()
    corrected[:, 1:] -= policy["class_logit_correction"]
    protected = [(start, end, 1.) for start, end in sorted(anchors)]
    if policy.get("algorithm") == "selective_whole_span":
        from name_selective_evidence import selective_spans

        return selective_spans(corrected, policy["addition_threshold"],
                               policy["removal_threshold"], anchors=protected)
    return evidence_spans(corrected, policy["cutoff"],
                          anchors=protected if policy["protect_anchors"] else ())


def retained_policy_anchors(logits, anchors, policy):
    if policy.get("algorithm") != "selective_whole_span":
        return set()
    from name_selective_evidence import all_o_posteriors

    corrected = logits.clone()
    corrected[:, 1:] -= policy["class_logit_correction"]
    ordered = sorted(anchors)
    outside = all_o_posteriors(corrected, ordered)
    return {span for span, probability in zip(ordered, outside, strict=True)
            if float(probability) <= policy["removal_threshold"]}


def validate_selected_decision(decision_path, development_path, checkpoint):
    decision, development = read_json(decision_path), read_json(development_path)
    if decision.get("status") != "selected":
        raise ValueError("Only a selected development decision may start live evaluation")
    if (development.get("scope") != "multisource_development_only"
            or development.get("source_changed_during_run") is not False
            or development.get("baseline_replay_verified") is not True
            or set(development.get("required_sources", [])) != set(REQUIRED_SOURCES)
            or decision.get("development_report_sha256") != digest(development_path)
            or decision.get("checker_sha256") != digest(
                ROOT / "scripts/name_multisource_selection.py")):
        raise ValueError("Selected decision lacks matching completed development provenance")
    recomputed = select_candidates(development["baselines"], development["candidates"],
                                   required_sources=development["required_sources"])
    if recomputed["status"] != "selected" or any(
            decision.get(key) != value for key, value in recomputed.items()):
        raise ValueError("Selected decision differs from recomputed strict selection")
    verify_hashes(development.get("frozen_sha256"))
    chosen = next(row for row in development["candidates"]
                  if row["id"] == decision["selected_id"])
    policy = chosen["policy"]
    validate_policy(policy)
    required = {"config.json", "head.safetensors", "manifest.json", "report.json",
                "training-text-hashes.json"}
    files = [path for path in checkpoint.rglob("*") if path.is_file()]
    if not required <= {path.name for path in files}:
        raise ValueError("Selected checkpoint is incomplete")
    artifact = {str(path.resolve()): digest(path) for path in files}
    if any(development["frozen_sha256"].get(path) != expected
           for path, expected in artifact.items()):
        raise ValueError("Checkpoint is not the exact artifact pinned by development caches")
    config, training = read_json(checkpoint / "config.json"), read_json(checkpoint / "report.json")
    manifest = read_json(checkpoint / "manifest.json")
    epochs = manifest.get("epochs")
    if (training.get("source_changed") is not False or "adaptation" in config
            or config.get("model_id") != MODEL_ID or config.get("revision") != MODEL_REVISION
            or config.get("selected_epoch") != training.get("selected_epoch")
            or config.get("threshold") != training.get("threshold")
            or type(epochs) is not int or epochs < 1
            or [row.get("epoch") for row in training.get("history", [])]
            != list(range(1, epochs + 1))):
        raise ValueError("Require a completed matching frozen-encoder checkpoint")
    return chosen, development, artifact


def strict_gate_passes(gate):
    return (isinstance(gate, dict) and gate.get("passed") is True
            and all(gate.get(key) == [] for key in REGRESSIONS))


def validate_live_reports(reports, identity):
    if len(reports) != 2 or {row.get("domain") for row in reports} != {"klue", "kdpii"}:
        raise ValueError("Heldout requires successful live development for both domains")
    for report in reports:
        if (report.get("scope") != "multisource_live_development"
                or report.get("passed") is not True
                or report.get("source_changed_during_run") is not False
                or report.get("cache_replay", {}).get("matched") is not True):
            raise ValueError("Live development must have passed exact replay and regression gates")
        for key, expected in identity.items():
            if report.get(key) != expected:
                raise ValueError(f"Live development differs from selected {key}")
        sources = report.get("sources", {})
        if (set(sources) != set(DOMAIN_SOURCES[report["domain"]])
                or any(not strict_gate_passes(row.get("regression_gate"))
                       or row.get("passed") is not True or row.get("aggregate_failures") != []
                       for row in sources.values())):
            raise ValueError("Live development source regression gate did not pass")
        verify_hashes(report.get("frozen_sha256"))
        consumed = Path(report["consumed_report"])
        if (report["frozen_sha256"].get(str(consumed.resolve())) != digest(consumed)
                or report.get("selected_ids") != read_json(consumed)["selected_ids"]):
            raise ValueError("Live development did not use the pinned consumed IDs")


def load_ancestry(checkpoint):
    """Recover normalized ancestry from pinned train/validation text only."""
    ancestry_path = checkpoint / "training-text-hashes.json"
    ancestry = read_json(ancestry_path)
    values = ancestry.get("text_sha256")
    if (ancestry.get("algorithm") != "sha256_utf8_exact_text" or not isinstance(values, list)
            or any(not isinstance(v, str) or len(v) != 64
                   or any(c not in "0123456789abcdef" for c in v) for v in values)):
        raise ValueError("Expected exact UTF-8 SHA256 ancestry")
    exact, normalized, found = set(values), set(), set()
    hashes = {str(ancestry_path.resolve()): digest(ancestry_path)}
    pending, visited, training_files = [checkpoint / "manifest.json"], set(), {}
    while pending:
        path = pending.pop().resolve()
        if path in visited:
            continue
        visited.add(path)
        hashes[str(path)] = digest(path)
        document = read_json(path)
        pins = {**document.get("input_sha256", {}), **document.get("source_sha256", {})}
        for filename, expected in pins.items():
            source = Path(filename)
            if source.name not in TRAINING_FILES and source.name != "manifest.json":
                continue
            if digest(source) != expected:
                raise ValueError(f"Pinned ancestry source changed: {source}")
            hashes[str(source.resolve())] = expected
            if source.name == "manifest.json":
                pending.append(source)
            else:
                training_files[str(source.resolve())] = source
    for path in training_files.values():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            original = hashlib.sha256(text.encode()).hexdigest()
            if original in exact:
                found.add(original)
                normalized.add(hashlib.sha256(normalize_text(text).text.encode()).hexdigest())
    if found != exact:
        raise ValueError("Cannot recover complete normalized ancestry from pinned training sources")
    return exact, normalized, hashes


def ancestry_overlaps(cases, exact, normalized):
    return dict(
        exact_ids=[row["id"] for row in cases
                   if hashlib.sha256(row["text"].encode()).hexdigest() in exact],
        normalized_ids=[row["id"] for row in cases if hashlib.sha256(
            normalize_text(row["text"]).text.encode()).hexdigest() in normalized],
    )


def read_cases(domain, mode, source, selected_ids):
    if domain == "klue":
        return read_selected_cases(source, selected_ids)
    if mode == "development":
        return [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    wanted = set(selected_ids)
    records = read_json(source)
    selected_rows = [row for row in records if row["sent_idx"] in wanted]
    selected = {row["sent_idx"]: row for row in selected_rows}
    if set(selected) != wanted or len(selected_rows) != len(selected):
        raise ValueError("Predeclared KDPII reserve IDs are absent; no replacement permitted")
    return [convert_record(selected[identifier]) for identifier in selected_ids]


def model_snapshot_hashes():
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    files = [path for path in snapshot.rglob("*") if path.is_file()]
    if not any(path.suffix == ".safetensors" for path in files):
        raise ValueError("Pinned local E5 snapshot has no safetensors weights")
    return {str(path.absolute()): digest(path) for path in files}


def runtime_signature():
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                packages={name: importlib.metadata.version(name)
                          for name in ("transformers", "presidio-analyzer", "safetensors")})


class TracedGuard:
    def __init__(self, guard):
        self.guard, self.findings = guard, []

    def analyze(self, text):
        findings = self.guard.analyze(text)
        self.findings.append([dict(entity=r.entity, start=r.start, end=r.end, score=r.score)
                              for r in findings])
        return findings

    def mask(self, text, *, style):
        return self.guard.mask(text, style=style)


def replay_comparison(cases, expected, expected_masks, expected_findings,
                      actual, actual_masks, actual_findings):
    if any(len(values) != len(cases) for values in (
            expected, expected_masks, expected_findings, actual, actual_masks, actual_findings)):
        raise ValueError("Replay comparisons require every original case")
    span_ids, mask_ids, score_cases, differences = [], [], 0, []
    for row, old, old_mask, old_details, new, new_mask, new_details in zip(
            cases, expected, expected_masks, expected_findings,
            actual, actual_masks, actual_findings, strict=True):
        before = {(r["entity"], r["start"], r["end"]): r["score"] for r in old_details}
        after = {(r["entity"], r["start"], r["end"]): r["score"] for r in new_details}
        if old != new or before.keys() != after.keys():
            span_ids.append(row["id"])
        if old_mask != new_mask:
            mask_ids.append(row["id"])
        values = [abs(before[key] - after[key]) for key in before.keys() & after.keys()
                  if before[key] != after[key]]
        if values and old == new and before.keys() == after.keys():
            score_cases += 1
        differences.extend(values)
    return dict(matched=not span_ids and not mask_ids, span_disagreement_ids=span_ids,
                mask_disagreement_ids=mask_ids, score_only_difference_cases=score_cases,
                score_difference_spans=len(differences),
                max_score_difference=max(differences, default=0.))


def source_groups(cases, domain):
    if domain == "klue":
        return {source: ([i for i, row in enumerate(cases) if row["id"].endswith("-" + source)],
                          [row for row in cases if row["id"].endswith("-" + source)])
                for source in DOMAIN_SOURCES[domain]}
    return {"kdpii/" + policy: (list(range(len(cases))), policy_cases(cases, policy))
            for policy in ("PS_NAME", "PS_NAME+PS_NICKNAME")}


def compare_sources(cases, domain, baseline, predictions, baseline_masks, masks):
    results = {}
    for source, (indices, rows) in source_groups(cases, domain).items():
        if not rows:
            raise ValueError(f"Required source has no rows: {source}")
        old, new = [baseline[i] for i in indices], [predictions[i] for i in indices]
        old_masks, new_masks = [baseline_masks[i] for i in indices], [masks[i] for i in indices]
        before, after = _score(rows, old, old_masks), _score(rows, new, new_masks)
        failures = [key for key in ("fully_covered_names", "f1") if after[key] < before[key]]
        failures += [key for key in ("negative_false_positive_sentences",
                                     "unnecessary_masked_characters") if after[key] > before[key]]
        regression = actual_mask_gate(rows, old, new, old_masks, new_masks)
        results[source] = dict(
            baseline=before, metrics=after, regression_gate=regression,
            aggregate_failures=failures, passed=not failures and strict_gate_passes(regression),
            bootstrap=paired_intervals(rows, old, new, old_masks, new_masks),
        )
    return results


def load_development_cache(path, development, domain, checkpoint, selected_ids):
    options = [path] if path else [Path(p) for p in development["frozen_sha256"]
                                   if p.endswith(".pt") and "development" in Path(p).name]
    matches = []
    for option in options:
        if development["frozen_sha256"].get(str(option.resolve())) != digest(option):
            raise ValueError("Development cache is not pinned by the selection report")
        payload = torch.load(option, map_location="cpu", weights_only=True)
        if (payload.get("scope") == "consumed_development_cache"
                and payload.get("domain") == domain
                and Path(payload.get("checkpoint", "")).resolve() == checkpoint.resolve()):
            matches.append(payload)
    if len(matches) != 1:
        raise ValueError("Require one pinned cache for this domain and checkpoint; use --cache")
    cache = matches[0]
    if cache.get("source_changed_during_run") is not False or cache["selected_ids"] != selected_ids:
        raise ValueError("Development cache differs from the consumed evaluation IDs")
    verify_hashes(cache.get("frozen_sha256"))
    for path in checkpoint.rglob("*"):
        if path.is_file() and cache["frozen_sha256"].get(str(path.resolve())) != digest(path):
            raise ValueError("Development cache used a different checkpoint artifact")
    return cache


def cached_replay(cases, cache, baseline_metrics, policy):
    validate_cache_cases(cache, cases)
    baseline = reconstruct(cases, baseline_metrics)
    baseline_guard = TracedGuard(guard_for(cases, [
        [RecognizerResult("KR_NAME", start, end, 1.) for start, end in spans]
        for spans in baseline]))
    _, old, old_masks = evaluate_guard(cases, baseline_guard)
    if old != baseline:
        raise ValueError("Baseline public replay changed frozen per-case coordinates")
    rows = []
    if len(cache.get("spans", [])) != len(cases):
        raise ValueError("Require every cached raw address row")
    for logits, original, anchors in zip(cache["logits"], cache["spans"], baseline, strict=True):
        selected = decode_policy(logits, anchors, policy)
        retained = retained_policy_anchors(logits, anchors, policy)
        addresses = [RecognizerResult(r["entity"], r["start"], r["end"], r["score"])
                     for r in original if r["entity"] == "KR_ADDRESS"]
        rows.append([*addresses, *[RecognizerResult("KR_NAME", start, end, score)
                                  for start, end, score in selected if (start, end) in retained
                                  or not any(
                                      start < address.end and address.start < end
                                      for address in addresses)]])
    candidate_guard = TracedGuard(guard_for(cases, rows))
    _, new, new_masks = evaluate_guard(cases, candidate_guard)
    return (old, old_masks, baseline_guard.findings), (new, new_masks, candidate_guard.findings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("development", "heldout"), required=True)
    parser.add_argument("--domain", choices=("klue", "kdpii"), required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reserve", type=Path,
                        default=ROOT / "benchmarks/results/name-multisource-v2-reserve.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--consumed-report", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--live-development", type=Path, nargs="+")
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(".manifest.json")
    if (args.output.resolve() == manifest_path.resolve()
            or args.output.exists() or manifest_path.exists()):
        parser.error("Require distinct unused output and manifest paths")
    chosen, development, artifact = validate_selected_decision(
        args.decision, args.development, args.checkpoint)
    identity = dict(decision_sha256=digest(args.decision),
                    development_report_sha256=digest(args.development),
                    artifact_sha256=artifact, selected_policy=chosen["policy"],
                    reserve_sha256=digest(args.reserve), device=args.device,
                    runtime_signature=runtime_signature())
    if args.mode == "heldout":
        validate_live_reports([read_json(path) for path in (args.live_development or [])], identity)
    reserve = read_json(args.reserve)
    if (reserve.get("scope") != "v2_evaluation_reserve_before_any_new_training"
            or reserve.get("test_labels_read") is not False):
        raise ValueError("Require the predeclared ID-only evaluation reserve")
    for domain, count in (("klue", 1000), ("kdpii", 2000)):
        ids = reserve[domain]["selected_ids"]
        if (len(ids) != count or len(set(ids)) != count
                or set(ids) & set(reserve[domain]["consumed_ids"])):
            raise ValueError("Reserve IDs changed, overlap consumed IDs, or require resampling")
    preparation = read_json(args.checkpoint / "data-manifest.json")
    exclusion_path = Path(reserve["exclusion_hashes_path"])
    if (preparation.get("input_sha256", {}).get(str(exclusion_path))
            != reserve.get("exclusion_hashes_sha256")
            or digest(exclusion_path) != reserve.get("exclusion_hashes_sha256")):
        raise ValueError("Reserve text exclusion differs from frozen training preparation")
    consumed_path = args.consumed_report or CONSUMED_REPORTS[args.domain]
    if development["frozen_sha256"].get(str(consumed_path.resolve())) != digest(consumed_path):
        raise ValueError("Consumed report is not pinned by development selection")
    consumed = read_json(consumed_path)
    selected_ids = (consumed["selected_ids"] if args.mode == "development"
                    else reserve[args.domain]["selected_ids"])
    if args.mode == "development" and len(selected_ids) != (1000 if args.domain == "klue" else 500):
        raise ValueError("Live development must use the original consumed population")
    source = args.source or Path(
        "/tmp/ko-pii-klue-ner/klue-ner-v1.1_dev.tsv" if args.domain == "klue" else
        "/tmp/ko-pii-kdpii-v1/" + ("cases.jsonl" if args.mode == "development" else "test.json"))
    source_pins = (consumed["frozen_sha256"] if args.mode == "development"
                   else reserve["source_sha256"])
    if source_pins.get(str(source.resolve())) != digest(source):
        raise ValueError("Source differs from the frozen consumed report or reserve")
    cache = (load_development_cache(args.cache, development, args.domain, args.checkpoint,
                                    selected_ids) if args.mode == "development" else None)
    exact, normalized, ancestry_hashes = load_ancestry(args.checkpoint)
    factory, factory_name = resolve_factory(chosen["policy"])
    hashes = {**development["frozen_sha256"], **artifact, **ancestry_hashes,
              **model_snapshot_hashes()}
    paths = [Path(__file__), args.decision, args.development, args.reserve, source, consumed_path,
             exclusion_path,
             *(args.live_development or []), *sorted((ROOT / "src/ko_pii_guard").glob("*.py"))]
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().is_relative_to(ROOT / "scripts"):
            paths.append(Path(filename))
    hashes.update({str(path.resolve()): digest(path) for path in paths})
    report = dict(scope="multisource_live_" + args.mode, domain=args.domain,
                  selected_id=chosen["id"], selected_ids=selected_ids,
                  consumed_report=str(consumed_path.resolve()), **identity,
                  frozen_sha256=hashes, runtime=dict(identity["runtime_signature"]),
                  candidate_factory=factory_name, model_kind="frozen",
                  baseline_model=dict(model_id=MODEL_ID, revision=MODEL_REVISION, threshold=.9),
                  guard_score_threshold=0., runtime_promotion=False, resampling=False,
                  limitations="Sentence bootstrap does not establish dialogue independence. "
                              "Upstream E5 training exposure is not proven absent.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    cases = read_cases(args.domain, args.mode, source, selected_ids)
    if [case["id"] for case in cases] != selected_ids or len(set(selected_ids)) != len(cases):
        raise ValueError("Cases differ from fixed selected IDs; no replacement permitted")
    if args.domain == "kdpii":
        validate_kdpii_cases(cases, selected_ids)
    overlap = ancestry_overlaps(cases, exact, normalized)
    report["ancestry_overlap"] = overlap
    if any(overlap.values()):
        changed = [path for path, expected in hashes.items()
                   if not Path(path).is_file() or digest(path) != expected]
        report.update(passed=False, evaluation_blocked="ancestry_text_overlap_no_resampling",
                      source_changed_during_run=bool(changed), changed_inputs=changed)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        raise SystemExit("Ancestry overlap; no inference or resampling")
    torch.set_num_threads(2)
    baseline_ner = KoreanNER.from_pretrained(device=args.device, score_threshold=.9)
    candidate_ner = factory(dict(checkpoint=str(args.checkpoint), model_kind="frozen",
                                 device=args.device, **chosen["policy"]))
    report["runtime"]["devices"] = validate_devices(baseline_ner, candidate_ner)
    guards = [TracedGuard(KoreanPIIGuard(entities=["KR_NAME"], score_threshold=0., ner=ner))
              for ner in (baseline_ner, candidate_ner)]
    outputs = [evaluate_guard(cases, guard, progress=lambda i, n, label=label: print(
        f"{label} {i}/{n}", flush=True)) for label, guard in zip(("baseline", "candidate"),
                                                               guards, strict=True)]
    (_, old, old_masks), (_, new, new_masks) = outputs
    sources = compare_sources(cases, args.domain, old, new, old_masks, new_masks)
    report["sources"] = sources
    replay_ok = True
    if args.mode == "development":
        baseline_metrics = consumed["baseline"]["metrics"]
        if args.domain == "kdpii":
            baseline_metrics = baseline_metrics["PS_NAME"]
        replay = cached_replay(cases, cache, baseline_metrics, chosen["policy"])
        comparisons = {label: replay_comparison(cases, *expected, predicted, masks, guard.findings)
                       for label, expected, (_, predicted, masks), guard in zip(
                           ("baseline", "candidate"), replay, outputs, guards, strict=True)}
        metric_differences = []
        for name, row in sources.items():
            for label, actual, expected in (
                    ("baseline", row["baseline"], development["baselines"][name]),
                    ("candidate", row["metrics"], chosen["sources"][name]["metrics"])):
                metric_differences.extend(dict(source=name, model=label, metric=key,
                                               expected=value, actual=actual.get(key))
                                          for key, value in compact(expected).items()
                                          if actual.get(key) != value)
        replay_ok = all(row["matched"] for row in comparisons.values()) and not metric_differences
        report["cache_replay"] = dict(matched=replay_ok, **comparisons,
                                      metric_differences=metric_differences)
    changed = [path for path, expected in hashes.items()
               if not Path(path).is_file() or digest(path) != expected]
    report.update(source_changed_during_run=bool(changed), changed_inputs=changed,
                  passed=replay_ok and all(row["passed"] for row in sources.values())
                  and not changed)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(passed=report["passed"], cache_replay_matched=replay_ok,
                          sources={name: dict(metrics=compact(row["metrics"]), passed=row["passed"])
                                   for name, row in sources.items()})), flush=True)
    if not report["passed"]:
        raise SystemExit("Live validation failed; heldout and runtime promotion remain blocked")


if __name__ == "__main__":
    main()
