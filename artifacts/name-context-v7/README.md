# Experimental Korean name rescue head v7

Unreleased, explicit opt-in. Preserves v6's character decoder and learned non-person
filter weights byte-for-byte. A separately trained character head proposes additional
name spans at confidence >=0.99; the existing contextual filter checks each proposal.
Only candidates disjoint from all accepted name/address spans are added. Existing
objects, positions and confidence values are preserved. Multi-window documents keep
the previous behavior. No inference name list or ordinary-word blacklist is used.

```python
from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard
from ko_pii_guard.ner import KoreanNER

guard = KoreanPIIGuard(
    entities=SUPPORTED_ENTITIES,
    ner=KoreanNER.from_pretrained(name_context_path="artifacts/name-context-v7"),
)
```

Requires the optional NER dependencies and explicitly downloaded pinned E5 backbone.
Inference is local and offline by default. Default E5 selection is unchanged. Added
head code/weights use the repository Apache-2.0 license; base-model terms also apply.

All nine known v6 missed-name cases pass exact-span and actual star-masking tests
(Red: 9 failures on v6; Green: 9 passes on v7). Full integration: 1,366 tests passed.

On the first new 192-sentence synthetic evaluation (234 names; 48 non-person cases),
v6 TP/FP/FN 224/4/10 becomes v7 228/4/6. Every one of 224 previously correct spans
is preserved and zero new false spans appear. Three existing non-person false-positive
sentences and six different missed names remain. This is not representative field
accuracy or a guarantee of error-free masking.

See [evaluation, provenance and reproduction](../../docs/name-context-v7-evaluation.md).
The manifest records settings and hashes before training; training_report.json records
selection (epoch 25) and first fresh evaluation. Original name/filter safe tensors are
retained beside the separate rescue_name_context.safetensors and versioned config.
Reproduce into a new output directory to preserve experiment history.
