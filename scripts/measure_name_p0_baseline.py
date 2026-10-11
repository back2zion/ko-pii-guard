"""Freeze the default E5 name baseline (P0-0) on every reproducible development set.

Only IDs already consumed as development data are opened. Reserve rows (KLUE 2,000 /
KDPII 2,391) are never selected, parsed or scored here. The model is the default
pinned E5 at name threshold 0.9; scoring uses public KoreanPIIGuard.analyze and a
separate actual mask(style="stars") call, as in the earlier frozen evaluators.

Usage: --dataset NAME --output PART.json  (one dataset per process), then
       --assemble PART.json ... --output benchmarks/results/name-p0-baseline-e5.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "benchmarks"))
RESULTS = ROOT / "benchmarks/results"
KLUE_TSV = Path("/tmp/ko-pii-klue-ner/klue-ner-v1.1_dev.tsv")
KDPII_TEST = Path("/tmp/ko-pii-kdpii-v1/test.json")


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def file_sha(path):
    return sha(Path(path).read_bytes())


def ids_from(name, key="selected_ids"):
    return json.loads((RESULTS / name).read_text())[key]


# name -> (loader kind, id list source file, id key, policies, id source description)
DATASETS = {
    "klue_dev_1000": ("klue", "name-span-v1-external.json", "selected_ids"),
    "klue_dev_999": ("klue", "name-generalization-final-v1-nonoverlap-heldout.manifest.json",
                     "selected_ids"),
    "klue_fresh_1000": ("klue", "name-multisource-v2-fresh-klue.manifest.json", "selected_ids"),
    "kdpii_test_500": ("kdpii500", "name-generalization-final-v1-nonoverlap-kdpii.manifest.json",
                       "selected_ids"),
    "kdpii_fresh_1992": ("kdpii_lossless", "name-multisource-v2-fresh-kdpii-lossless.manifest.json",
                         "selected_ids"),
    "synthetic_v11_192": ("synthetic", None, None),
}


def load_cases(name):
    from evaluate_name_generalization import read_selected_cases

    kind, file, key = DATASETS[name]
    if kind == "klue":
        ids = ids_from(file, key)
        return read_selected_cases(KLUE_TSV, ids), ids, file
    if kind == "kdpii500":
        ids = ids_from(file, key)
        cases = [json.loads(line) for line in
                 Path("/tmp/ko-pii-kdpii-v1/cases.jsonl").read_text().splitlines()]
        if [c["id"] for c in cases] != ids:
            raise ValueError("KDPII 500 cases differ from the consumed ID list")
        return cases, ids, file
    if kind == "kdpii_lossless":
        import kdpii_lossless_annotations as lossless

        ids = ids_from(file, key)
        cases, _ = lossless.read_cases_lossless(KDPII_TEST, ids)
        return cases, ids, file
    from name_address_benchmark import load_cases as load

    path = ROOT / "benchmarks/data/name_context_v11.jsonl"
    cases = [c for c in load(path) if c["split"] == "evaluation"]
    return cases, [c["id"] for c in cases], str(path.relative_to(ROOT))


def source_of(name, case):
    if name.startswith("klue"):
        return case["id"].rsplit("-", 1)[-1]
    if name.startswith("kdpii"):
        return "kdpii_v1_test"
    return "synthetic_v11_evaluation"


def run_dataset(name, device):
    import torch
    import transformers
    from evaluate_name_generalization import _score
    from train_name_span_experiment import gold

    from ko_pii_guard import KoreanPIIGuard
    from ko_pii_guard.ner import MODEL_ID, MODEL_REVISION, KoreanNER

    torch.set_num_threads(2)
    cases, ids, id_source = load_cases(name)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate IDs")
    ner = KoreanNER.from_pretrained(device=device, score_threshold=0.9)
    guard = KoreanPIIGuard(entities=["KR_NAME"], ner=ner, score_threshold=0.0)
    predictions, masks, started = [], [], time.time()
    for index, case in enumerate(cases, 1):
        predictions.append({(r.start, r.end) for r in guard.analyze(case["text"])
                            if r.entity == "KR_NAME"})
        masked = guard.mask(case["text"], style="stars")
        if len(masked) != len(case["text"]):
            raise ValueError("stars mask changed coordinates")
        masks.append(masked)
        if index % 100 == 0 or index == len(cases):
            print(name, index, len(cases), flush=True)
    elapsed = time.time() - started

    def score(subset, policy_field=None):
        sel = [i for i in range(len(cases)) if i in subset]
        sub = [dict(cases[i], expected=cases[i][policy_field]) if policy_field else cases[i]
               for i in sel]
        result = _score(sub, [predictions[i] for i in sel], [masks[i] for i in sel])
        result.pop("errors")
        return result

    everything = set(range(len(cases)))
    policies = {"PS_NAME": None, "PS_NAME+PS_NICKNAME": "alternate_expected"} \
        if name.startswith("kdpii") else {"primary": None}
    groups = {}
    for source in sorted({source_of(name, c) for c in cases}):
        groups[source] = {i for i, c in enumerate(cases) if source_of(name, c) == source}
    metrics = {}
    for policy, field in policies.items():
        metrics[policy] = dict(overall=score(everything, field),
                               by_source={s: score(g, field) for s, g in groups.items()})
    per_case = []
    for case, spans, masked in zip(cases, predictions, masks, strict=True):
        row = dict(id=case["id"], source=source_of(name, case),
                   predicted=sorted(spans), text_sha256=sha(case["text"]))
        for policy, field in policies.items():
            expected = {(e["start"], e["end"]) for e in case[field or "expected"]
                        if e["entity"] == "KR_NAME"}
            row[f"gold[{policy}]"] = sorted(expected)
            row[f"fully_masked[{policy}]"] = sorted(
                (s, e) for s, e in expected if masked[s:e] == "*" * (e - s))
        per_case.append(row)
    del gold
    return dict(
        dataset=name, id_list_source=id_source, sentences=len(cases),
        ids_sha256=sha("\n".join(ids)),
        texts_sha256=sha("\n".join(sha(c["text"]) for c in cases)),
        id_text_pairs_sha256=sha("\n".join(f"{c['id']}\t{sha(c['text'])}" for c in cases)),
        model=dict(model_id=MODEL_ID, revision=MODEL_REVISION, name_threshold=0.9,
                   guard_score_threshold=0.0),
        pipeline="Public KoreanPIIGuard(entities=['KR_NAME'], score_threshold=0.0).analyze "
                 "and separate mask(style='stars'); exact (start,end) span match",
        runtime=dict(python=platform.python_version(), torch=torch.__version__,
                     transformers=transformers.__version__, device=device,
                     cuda_available=torch.cuda.is_available(), torch_threads=2,
                     platform=platform.platform()),
        wall_seconds=round(elapsed, 1), metrics=metrics, per_case=per_case,
    )


def execution_conditions():
    import subprocess

    from ko_pii_guard.ner import MODEL_ID, MODEL_REVISION, KoreanNER

    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                              text=True).stdout.strip()

    ner = KoreanNER.from_pretrained(device="cpu", score_threshold=0.9)
    snapshot = Path.home() / ".cache/huggingface/hub/models--FrameByFrame--korean-pii-e5-base" \
        / "snapshots" / MODEL_REVISION
    files = {p.name: file_sha(p.resolve()) for p in sorted(snapshot.iterdir())}
    return dict(
        code_commit=git("rev-parse", "HEAD"), working_tree_status=git("status", "--short"),
        model_id=MODEL_ID, model_revision=MODEL_REVISION, hf_cache_snapshot=str(snapshot),
        snapshot_file_sha256=files, tokenizer=type(ner.tokenizer).__name__,
        id2label=ner.model.config.id2label, name_threshold=ner.score_threshold,
        max_window_tokens=ner.max_length, window_stride=ner.stride, batch_size=ner.batch_size,
        long_documents="overlapping token windows (return_overflowing_tokens, stride); "
                       "overlap tokens take the window with more two-sided context",
        label_mapping="KLUE PS -> KR_NAME (other labels non-person); KDPII PS_NAME -> KR_NAME, "
                      "PS_NICKNAME only in the alternate policy; synthetic v11 KR_NAME",
        postprocessing="KoreanPIIGuard.analyze over KoreanNER spans (src/ko_pii_guard frozen; "
                       "hashes in frozen_sha256); scoring is exact (start,end) over original "
                       "string coordinates, entity KR_NAME only, guard score_threshold 0.0",
        raw_vs_postprocessed="Main metrics are guard-level (post-processed). Raw KoreanNER.analyze "
                             "metrics are in raw_ner_secondary and are never mixed in.",
        k_pii_bench="not used for name measurement (names outside its default profile); "
                    "pinned revision 844f92f7beba98fa964f957ba324017e82451701 per docs/evaluation.md",
    )


DOCUMENTED = {  # numbers recorded in docs/name-p0-research-plan.md section 1.1
    ("klue_dev_1000", "primary", "__all__"): dict(tp=765, fp=126, fn=143, fully_covered_names=812, f1=0.8505),
    ("klue_dev_999", "primary", "__all__"): dict(f1=0.8458, fp=138, fn=133, mask_failures=80),
    ("kdpii_test_500", "PS_NAME", "__all__"): dict(f1=0.8571),
    ("kdpii_test_500", "PS_NAME+PS_NICKNAME", "__all__"): dict(f1=0.6122),
    ("klue_fresh_1000", "primary", "nsmc"): dict(f1=0.8058),
    ("klue_fresh_1000", "primary", "wikitree"): dict(f1=0.8844),
    ("kdpii_fresh_1992", "PS_NAME", "__all__"): dict(f1=0.8571),
    ("kdpii_fresh_1992", "PS_NAME+PS_NICKNAME", "__all__"): dict(f1=0.7128),
}


def documented_comparison(datasets):
    rows = []
    for (name, policy, source), documented in DOCUMENTED.items():
        m = datasets[name]["metrics"][policy]
        m = m["overall"] if source == "__all__" else m["by_source"][source]
        measured = {}
        for key, value in documented.items():
            if key == "mask_failures":
                got = m["gold_names"] - m["fully_covered_names"]
            else:
                got = m[key]
            measured[key] = got
        match = all((abs(measured[k] - v) < 5e-5 if isinstance(v, float) else measured[k] == v)
                    for k, v in documented.items())
        rows.append(dict(dataset=name, policy=policy, source=source, documented=documented,
                         measured=measured, matches=match,
                         f1_matched_at="4 significant digits (percent to 2 decimals)"))
    return rows


def assemble(parts, output, raw=()):
    loaded = [json.loads(Path(p).read_text()) for p in parts]
    datasets, side = {}, []
    for part in loaded:
        for row in part.pop("per_case"):
            side.append(dict(dataset=part["dataset"], **row))
        datasets[part["dataset"]] = part
    # Source-level pooling: KLUE NSMC/Wikitree across the three consumed KLUE development sets.
    pooled = {}
    for source in ("nsmc", "wikitree"):
        rows = [d["metrics"]["primary"]["by_source"][source] for n, d in datasets.items()
                if n.startswith("klue") and source in d["metrics"]["primary"]["by_source"]]
        keys = ("tp", "fp", "fn", "fully_covered_names", "unnecessary_masked_characters",
                "negative_false_positive_sentences", "negative_sentences", "sentences",
                "gold_names")
        total = {k: sum(r[k] for r in rows) for k in keys}
        total["f1"] = 2 * total["tp"] / (2 * total["tp"] + total["fp"] + total["fn"])
        pooled[source] = dict(total, pooled_from=[n for n in datasets if n.startswith("klue")])
    sidecar = output.with_suffix(".predictions.jsonl")
    conditions = execution_conditions()
    if output.exists() or sidecar.exists():
        raise ValueError("Refusing to overwrite frozen baseline")
    sidecar.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in side))
    report = dict(
        scope="name_p0_baseline_e5_frozen", milestone="M0 / P0-0",
        created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        status_of_data="development only (all consumed); no independent heldout exists",
        reserve_unopened_confirmation="KLUE 2,000 and KDPII 2,391 reserve rows were not selected, "
                                      "parsed or scored.",
        documented_comparison=documented_comparison(datasets),
        execution_conditions=conditions,
        raw_ner_secondary={json.loads(Path(r).read_text())['dataset']: json.loads(Path(r).read_text())
                           for r in raw},
        datasets=datasets, pooled_klue_by_source=pooled,
        predictions_sidecar=dict(path=str(sidecar.relative_to(ROOT)), sha256=file_sha(sidecar),
                                 rows=len(side),
                                 note="per-sentence predicted spans, gold spans and fully-masked "
                                      "spans (no text) for later per-span regression checks"),
        script_sha256=file_sha(__file__),
        frozen_sha256={str(p): file_sha(p) for p in (
            KLUE_TSV, KDPII_TEST, ROOT / "scripts/evaluate_name_generalization.py",
            ROOT / "scripts/train_name_span_experiment.py",
            ROOT / "scripts/kdpii_lossless_annotations.py",
            ROOT / "scripts/fetch_kdpii_evaluation.py",
            ROOT / "benchmarks/data/name_context_v11.jsonl",
            *sorted((ROOT / "src/ko_pii_guard").glob("*.py")))},
    )
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS))
    parser.add_argument("--assemble", nargs="+")
    parser.add_argument("--raw", nargs="*", default=[])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.assemble:
        assemble(args.assemble, args.output, args.raw)
        return
    if args.output.exists():
        parser.error("do not overwrite")
    args.output.write_text(json.dumps(run_dataset(args.dataset, args.device),
                                      ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
