# Name context head v2

This checkpoint repairs the known v1 misses and regressions through broader
supervised character-level training. It is explicitly selected; the pinned E5
backbone remains a separate local dependency. No external inference API is used.

```python
from ko_pii_guard import KoreanPIIGuard, SUPPORTED_ENTITIES
from ko_pii_guard.ner import KoreanNER

ner = KoreanNER.from_pretrained(name_context_path="artifacts/name-context-v2")
guard = KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=ner)
print(guard.mask("첨부: 김철수_이력서.pdf"))
# 첨부: <KR_NAME>_이력서.pdf
```

Only the small name decoder is distributed here. Structured identifier rules,
E5 address inference, offset conventions, and caller masking policies are retained.
The v1 artifact and its original measurements are preserved for comparison.

## Training and selection

The v2 corpus was authored and frozen before training. It has 4,488 training,
168 validation, and 232 evaluation sentences. All gold target name strings
(including second/repeated people), full sentences and context IDs are disjoint
between splits. The inspected v1 splits and previous development corpora are now
training/selection data, not new independent evaluation data.

Training adds single-character names, common-word names, filename mentions,
multiple people, repeated mentions and ordinary-word/file-title negatives. The E5
backbone and the character-head architecture remain unchanged. No runtime name
whitelist, word blacklist, unconditional baseline fallback or particle stripping
was introduced to repair these failures.

Checkpoint selection first minimizes errors on the declared known development
regressions, then maximizes validation exact-span F1, then minimizes validation
false positives (earliest tie). The evaluation split is run only after selection.
All 50 epochs, settings, source/data/checkpoint hashes and results are retained in
`training_manifest.json` and `training_report.json`.

Run from the repository root with the locked optional `ner` environment and the
explicitly downloaded pinned E5 model:

```bash
uv run --frozen python benchmarks/name_context_v2_cases.py --check
uv run --frozen python scripts/train_name_context.py --data benchmarks/data/name_context_v2.jsonl --epochs 50 --regression-data benchmarks/data/name_context_training.jsonl --regression-data benchmarks/data/name_address.jsonl --regression-data benchmarks/data/business_korean.jsonl --regression-data benchmarks/data/name_field_boundaries.jsonl --output /tmp/name-context-v2-reproduction
uv run --frozen python benchmarks/name_context_v2_benchmark.py --checkpoint artifacts/name-context-v2 --output /tmp/name-context-v2-acceptance.json --check
KO_PII_TEST_NAME_CONTEXT=1 uv run --frozen pytest -q tests/test_name_context_head.py
```

Use a fresh output directory. The training script refuses to overwrite previous
experiments. Floating-point byte identity across different hardware/software
versions is not promised.

## Measurement and scope

[Evaluation and acceptance results](../../docs/name-context-v2-evaluation.md)
distinguish known-error regression checks from the new evaluation split. These
are authored synthetic crossed templates, not independent field samples or a
population-level accuracy estimate. Correcting known failures does not guarantee
that all real-world name ambiguities, files or long documents will be correct.

## Licensing

Code, directly authored synthetic annotations and the newly trained head use the
repository's Apache-2.0 license. E5 weights are not included. The separately
cached base is `FrameByFrame/korean-pii-e5-base`, pinned revision
`a308c54b4407819624a5661e31e162a269f39818`, with its own MIT notices. Preserve those
notices when distributing E5. See [base model provenance](../../docs/name-address-sources.md)
and the [recent research review](../../docs/name-context-research.md). This is not
an implementation or reproduction of the reviewed papers' models or datasets.
