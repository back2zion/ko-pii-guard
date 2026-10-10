# Experimental Korean name context head v4

**Rejected for adoption:** on the new v4 evaluation, person misses increased
from v3's 24 to 32 despite negative-sentence errors dropping from 6/60 to 1/60.
Eight previously correct person sentences now have explicit regression tests.
This checkpoint is retained only as reproducible experiment history.

This optional character BIOES head is fine-tuned from v3 over the same frozen,
pinned `FrameByFrame/korean-pii-e5-base` backbone. It remains an unreleased,
explicitly selected experiment under this repository's Apache-2.0 license.

```python
from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard
from ko_pii_guard.ner import KoreanNER

guard = KoreanPIIGuard(
    entities=SUPPORTED_ENTITIES,
    ner=KoreanNER.from_pretrained(name_context_path="artifacts/name-context-v4"),
)
```

The optional NER dependencies and explicitly downloaded pinned E5 backbone are
required; inference is local and offline by default. Default model behavior is
unchanged unless this checkpoint is selected. No ordinary-word blacklist, name
dictionary, blanket title/file veto or sentence-level name suppression is added
to inference.

All 40 retired v3 negative sentences are regression tests, and 163 development
title/author contrasts protect genuine person names in the same sentence.
V3 data is explicitly retired into development. V4's separately named validation
and evaluation splits use disjoint gold names, literal sentences and context IDs.
The new evaluation is synthetic evidence, not representative field accuracy.

See [reproduction and measured results](../../docs/name-context-v4-evaluation.md).
The manifest records initialization, source and data hashes before training.
The training report records all epochs, checkpoint selection and the first new
evaluation. Earlier checkpoint versions and their original reports are preserved.
