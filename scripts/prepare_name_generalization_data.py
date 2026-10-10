"""Prepare pinned KLUE *train* plus existing synthetic train for name research.

Never reads KLUE dev. Pass already inspected evaluation text hashes separately.
Original text and character offsets are preserved; raw data stays in /tmp by default.
KLUE content is CC-BY-SA-4.0, separately from this repository's code license.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "3efd98708a40ff49251fddde35453f8fbb11f536"
TRAIN_SHA256 = "34b9d3d9f9ce9e064abc6ba4c27af43b4b70f2d6cde62c076a28e8aaa17cc544"
FILES = {
    "klue_benchmark/klue-ner-v1.1/klue-ner-v1.1_train.tsv": TRAIN_SHA256,
    "License.md": "7abe19ec9bb73b36141b999b861d24ad855e808bafe0f81e84cce28556f6c297",
}
TRAIN_ID = re.compile(r"klue-ner-v1_train_\d+_(wikitree|nsmc)$")
TAGS = {"O", *[f"{p}-{t}" for p in ("B", "I") for t in ("PS", "OG", "LC", "DT", "TI", "QT")]}


def text_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verified_bytes(path, url, expected):
    if path.exists():
        content = path.read_bytes()
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "ko-pii-guard-research"})
        with urllib.request.urlopen(request, timeout=45) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"Pinned source hash mismatch: {path.name}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return content


def parse_train(content):
    cases, chars, tags, identifier = [], [], [], None

    def flush():
        if identifier is None:
            return
        if not chars:
            raise ValueError(f"Empty source document: {identifier}")
        expected, active = [], None
        for i, tag in enumerate([*tags, "O"]):
            if active is not None and tag != "I-PS":
                expected.append(dict(entity="KR_NAME", start=active, end=i))
                active = None
            if tag == "B-PS":
                active = i
        cases.append(
            dict(
                id=identifier,
                text="".join(chars),
                expected=expected,
                source=TRAIN_ID.fullmatch(identifier).group(1),
                split="train",
                char_bio=tags[:],
                license="CC-BY-SA-4.0",
                track="real_klue_train",
            )
        )

    for line in content.splitlines():
        if line.startswith("## klue-ner-"):
            flush()
            identifier = line[3:].split("\t", 1)[0]
            if not TRAIN_ID.fullmatch(identifier):
                raise ValueError(f"Only official KLUE train IDs are allowed: {identifier}")
            chars, tags = [], []
        elif line.startswith("##") or not line:
            continue
        else:
            if identifier is None:
                raise ValueError("Character annotation before document header")
            char, tag = line.rsplit("\t", 1)
            if len(char) != 1 or tag not in TAGS:
                raise ValueError(f"Invalid character annotation: {identifier}")
            if tag.startswith("I-") and (not tags or tags[-1] not in ("B-" + tag[2:], tag)):
                raise ValueError(f"Orphan inside BIO tag: {identifier}")
            chars.append(char)
            tags.append(tag)
    flush()
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Empty corpus or duplicate document ID")
    return cases


def name_spans(case):
    spans = sorted((e["start"], e["end"]) for e in case["expected"] if e["entity"] == "KR_NAME")
    previous_end = 0
    for start, end in spans:
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("Name span coordinates must be integers")
        if not previous_end <= start < end <= len(case["text"]):
            raise ValueError(f"Invalid or overlapping name span: {case['id']}")
        previous_end = end
    return tuple(spans)


def normalized(case, *, source=None, split=None):
    tags = ["O"] * len(case["text"])
    spans = name_spans(case)
    for start, end in spans:
        tags[start] = "B-PS"
        tags[start + 1 : end] = ["I-PS"] * (end - start - 1)
    return dict(
        id=case["id"],
        text=case["text"],
        expected=[dict(entity="KR_NAME", start=s, end=e) for s, e in spans],
        source=source or case["source"],
        split=split or case["split"],
        char_bio=case.get("char_bio", tags),
        char_name_bio=tags,
        license=case.get("license", "Apache-2.0"),
    )


def prepare(real, synthetic, *, excluded_hashes=(), validation_fraction=0.1, seed=20261010):
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    ids = [c["id"] for c in [*real, *synthetic]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate document ID across input corpora")
    excluded = set(excluded_hashes)
    excluded.update(text_digest(c["text"]) for c in synthetic if c["split"] != "train")
    candidates = [normalized(c) for c in real]
    candidates += [normalized(c, source="synthetic") for c in synthetic if c["split"] == "train"]
    groups = defaultdict(list)
    excluded_ids = []
    for case in candidates:
        if text_digest(case["text"]) in excluded:
            excluded_ids.append(case["id"])
        else:
            groups[case["text"]].append(case)
    unique, conflicting, removed = [], [], 0
    for group in groups.values():
        # Keep real supervision ahead of a synthetic copy; tie-break only by ID.
        group.sort(key=lambda c: (c["source"] == "synthetic", c["id"]))
        if len({tuple(c["char_bio"]) for c in group}) > 1:
            conflicting.append(sorted(c["id"] for c in group))
            continue
        unique.append(group[0])
        removed += len(group) - 1
    by_source = defaultdict(list)
    for case in unique:
        by_source[case["source"]].append(case)
    train, validation = [], []
    for source, group in sorted(by_source.items()):
        group.sort(key=lambda c: (text_digest(f"ko-pii-real-train-v1:{seed}:{c['id']}"), c["id"]))
        n_validation = int(len(group) * validation_fraction) if source != "synthetic" else 0
        validation.extend(dict(c, split="validation") for c in group[:n_validation])
        train.extend(dict(c, split="train") for c in group[n_validation:])
    train.sort(key=lambda c: c["id"])
    validation.sort(key=lambda c: c["id"])
    assert not {c["text"] for c in train} & {c["text"] for c in validation}
    report = dict(
        original_real_rows=len(real),
        original_synthetic_rows=len(synthetic),
        synthetic_nontraining_rows_excluded=sum(c["split"] != "train" for c in synthetic),
        excluded_text_rows=len(excluded_ids),
        excluded_ids=sorted(excluded_ids),
        duplicate_rows_removed=removed,
        conflicting_text_groups=sorted(conflicting),
        conflicting_rows_removed=sum(map(len, conflicting)),
        split_counts={
            "train": dict(Counter(c["source"] for c in train)),
            "validation": dict(Counter(c["source"] for c in validation)),
        },
        cross_split_exact_text_overlap=0,
    )
    return train, validation, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/ko-pii-klue-name-train"))
    parser.add_argument(
        "--synthetic", type=Path, default=ROOT / "benchmarks/data/name_context_v11.jsonl"
    )
    parser.add_argument("--exclude-text-hashes", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/tmp/ko-pii-name-generalization-v1"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new or empty output directory; never overwrite prepared data")
    source = dict(
        repository="https://github.com/KLUE-benchmark/KLUE",
        revision=REVISION,
        license="CC-BY-SA-4.0",
        files={},
    )
    for relative, expected in FILES.items():
        url = f"https://raw.githubusercontent.com/KLUE-benchmark/KLUE/{REVISION}/{relative}"
        content = verified_bytes(args.source_dir / Path(relative).name, url, expected)
        source["files"][Path(relative).name] = dict(url=url, sha256=expected, bytes=len(content))
    exclusions = json.loads(args.exclude_text_hashes.read_text())
    if not isinstance(exclusions, list) or not all(
        isinstance(v, str) and re.fullmatch(r"[a-f0-9]{64}", v) for v in exclusions
    ):
        raise ValueError("Exclusion file must contain a JSON list of SHA256 text hashes")
    real = parse_train((args.source_dir / "klue-ner-v1.1_train.tsv").read_text())
    synthetic = [json.loads(line) for line in args.synthetic.read_text().splitlines() if line]
    train, validation, report = prepare(real, synthetic, excluded_hashes=exclusions)
    validation_texts = {c["text"] for c in validation}
    synthetic_validation = [
        normalized(c, source="synthetic")
        for c in synthetic
        if c["split"] == "validation" and c["text"] not in validation_texts
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for filename, cases in [
        ("train.jsonl", train),
        ("validation.jsonl", validation),
        ("synthetic-validation.jsonl", synthetic_validation),
    ]:
        data = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases).encode()
        (args.output / filename).write_bytes(data)
        outputs[filename] = dict(rows=len(cases), sha256=hashlib.sha256(data).hexdigest())
    manifest = dict(
        source=source,
        preparation=report,
        outputs=outputs,
        input_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), args.synthetic, args.exclude_text_hashes)
        },
        selection="Exact-text groups; source-stratified first floor(10%) by SHA256 of seed+ID",
        seed=20261010,
        validation_fraction=0.1,
        validation_scope="Previously published KLUE training data; base E5 exposure is expected",
        original_offsets_preserved=True,
        development_corpus_read=False,
        excluded_evaluation_text_hashes=len(set(exclusions)),
        limitation="Exact duplicates removed; no original article/thread grouping is available",
        attribution="KLUE: Korean Language Understanding Evaluation (Park et al., NeurIPS 2021)",
        license="KLUE-derived data CC-BY-SA-4.0; synthetic-only source Apache-2.0",
        source_license_file=str(args.source_dir / "License.md"),
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), outputs=outputs, preparation=report), indent=2))


if __name__ == "__main__":
    main()
