# Experimental Korean name context head v3

This optional character BIOES head uses the frozen, pinned
`FrameByFrame/korean-pii-e5-base` backbone. It is an unreleased experiment under
the repository's Apache-2.0 license. Existing v1/v2 checkpoints are preserved.

```python
from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard
from ko_pii_guard.ner import KoreanNER

guard = KoreanPIIGuard(
    entities=SUPPORTED_ENTITIES,
    ner=KoreanNER.from_pretrained(name_context_path="artifacts/name-context-v3"),
)
```

The optional NER dependencies and explicitly downloaded pinned E5 backbone are
required. Inference remains local and offline by default. No target name list
or ordinary-word blacklist is added to inference. The default E5 behavior is
unchanged unless this checkpoint is explicitly selected.

The 26 v2 error sentences were fixed as tests before training v3. All inspected
v2 splits are now development data. New validation and evaluation names, literal
sentences and context IDs are disjoint from training and each other. The new
evaluation is synthetic engineering evidence, not representative field accuracy.

See [the evaluation and reproduction commands](../../docs/name-context-v3-evaluation.md).
`training_manifest.json` freezes data/source hashes and settings before training;
`training_report.json` records selection history and the first new evaluation.
The safe tensor weights require the matching versioned configuration and pinned
backbone. The original checkpoint paths should not be overwritten when rerunning
training.

Measured results: all 26 previously failing v2 sentences now pass; the public
v2 development evaluation is 300 TP / 0 FP / 0 FN. On the new 280-sentence
evaluation (400 names, 40 negatives), v3 is 395 TP / 31 FP / 5 FN, with 395 fully
masked spans. On that same new data, v2 is 368 TP / 31 FP / 32 FN. Negative
sentences with false positives increase from 10/40 to 12/40; v3 is not an
across-the-board quality improvement. The full v3 integration suite passes 999
tests. These results do not establish error-free masking on arbitrary text.
