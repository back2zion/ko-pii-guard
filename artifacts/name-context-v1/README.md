# Experimental name context head v1

Historical checkpoint. The known 21 misses and seven regressions are addressed
by [v2](../name-context-v2/README.md); retain this artifact and its original
reports for reproducible comparison.

**Explicit opt-in; not the default model.** Trained locally on 2026-10-10 with a
frozen, pinned E5 backbone. Only the 748 KiB character decoder is included here;
E5 weights must already be present in the local Hugging Face cache.

```python
from ko_pii_guard import KoreanPIIGuard, SUPPORTED_ENTITIES
from ko_pii_guard.ner import KoreanNER

ner = KoreanNER.from_pretrained(name_context_path="artifacts/name-context-v1")
guard = KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=ner)
print(guard.mask("간호사는 대기 중인 환자 사공이든 씨를 진료실로 안내했다."))
# 간호사는 대기 중인 환자 <KR_NAME> 씨를 진료실로 안내했다.
```

The head replaces E5 name predictions; it keeps E5 address predictions and all
structured identifier recognizers. It can generate previously missed names and
split name boundaries inside a tokenizer token. It does not run an external API,
use a name dictionary, or strip Korean particles by spelling.

## Reproduce

Install the repository's `ner` and `dev` extras, explicitly download the pinned E5
model as documented, then run from the repository root:

```bash
uv run --frozen python benchmarks/name_context_training_cases.py --check
uv run --frozen python scripts/train_name_context.py --output /tmp/name-context-reproduction
uv run --frozen python benchmarks/name_context_head_benchmark.py --checkpoint artifacts/name-context-v1 --output /tmp/name-context-regressions.json
KO_PII_TEST_NAME_CONTEXT=1 uv run --frozen pytest -q tests/test_name_context_head.py
```

Use a new output directory: the training script refuses to overwrite experiments.
The manifest was written before inference and records the data/source hashes,
fixed architecture, seed, settings, and selection rule. `training_report.json`
records all 25 epochs, the selected epoch 10, checkpoint hash, and public-API
held-out evaluation. CPU thread count was 2; byte-identical floating-point results
across different hardware/dependency versions are not promised.

## Scope and limitations

Training/validation/evaluation have 1,758/80/140 synthetic sentences. Target names,
full sentences, and context IDs are disjoint across splits. The 808 historical
name-context cases are now training data, not held-out evidence. The new 140-case
evaluation has 128 names and only 12 negative sentences. These crossed templates
are not independent real-world samples; no population accuracy claim is made.

On that evaluation, exact spans were 91/128 for E5 and 107/128 for this head.
Incorrect predicted spans fell from 23 to 0; full-name masking rose from 98/128
to 107/128. Negative-sentence false positives were 0/12 for both. The new head
still misses 21 names, and some cases correct under E5 regress. Separate existing
development checks also reveal missing names in filenames and incomplete
multi-person recognition. Do not infer broad superiority or use it as the default.

The four previously documented context/boundary failures are corrected in a real
model regression test. These were included in training and are regression evidence,
not proof of generalization. Review [research notes](../../docs/name-context-research.md)
and the [latest development comparison](../../benchmarks/results/name-context-head-regressions-v2.json).

## Provenance and licensing

The decoder code, directly authored synthetic training data, and newly trained
head are provided under this repository's Apache-2.0 license. E5 is a separate
third-party model: `FrameByFrame/korean-pii-e5-base`, pinned revision
`a308c54b4407819624a5661e31e162a269f39818`. Its weights are not included here.
Preserve the separately supplied E5 MIT notices when distributing E5 weights.
See [base model sources](../../docs/name-address-sources.md) for its stated
upstream training-data provenance. No datasets or weights from the recently
reviewed papers are redistributed by this experiment.
