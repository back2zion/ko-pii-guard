# Experimental Korean name decoder ensemble v9

Unreleased and explicit opt-in. The v7 primary and rescue weights are byte-identical;
v8's refined rescue is an additional decoder and its learned non-person filter is
used for candidates. Old accepted rescue spans take priority over overlapping new
proposals. No name exception list or common-word blacklist is used in inference.
Multi-window long documents retain previous behavior. Extra short-document inference
work is required for the additional decoder.

```python
from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard
from ko_pii_guard.ner import KoreanNER

guard = KoreanPIIGuard(
    entities=SUPPORTED_ENTITIES,
    ner=KoreanNER.from_pretrained(name_context_path="artifacts/name-context-v9"),
)
```

Requires optional NER dependencies and the explicitly downloaded pinned E5 backbone.
Inference is local and offline by default; default E5 selection is unchanged. Added
code/weights use repository Apache-2.0; underlying model terms also apply.

All six known missed names and four false spans from the v7 evaluation are fixed.
The original 192 sentences now score TP234/FP0/FN0; the earlier v6 evaluation retains
the same zero-error counts. The failed v8 regression is also fixed as a test.
Full selected-model integration: 1,381 tests passed.

Fresh synthetic evaluation (192 sentences, 234 names, 48 negatives): v7 TP/FP/FN
228/3/6 becomes v9 229/2/5. All 228 previously correct spans are preserved and no new
false span is introduced. Five different misses and two existing false spans remain.
This does not establish error-free masking or representative field accuracy.

See [evaluation, rejection history and reproduction](../../docs/name-context-v9-evaluation.md).
assembly_manifest.json freezes parents, code and data before evaluation;
assembly_report.json records known development gates and the first independent test.
The parent v8 is a rejected standalone experiment; its refined decoder is added beside
v7's preserved rescue here. Reproduce into a new output directory.
