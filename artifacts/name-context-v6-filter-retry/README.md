# Experimental Korean name candidate filter v6

Opt-in, unreleased, local inference. Uses the pinned E5 backbone and the byte-identical
v3 character decoder, plus a learned non-person candidate filter. Candidate/global
E5 representations and the preceding/following 16-character context inform the filter.
No inference name whitelist or common-word blacklist is used. Multi-window documents
retain the existing decoder outputs without the new filter.

```python
from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard
from ko_pii_guard.ner import KoreanNER

guard = KoreanPIIGuard(
    entities=SUPPORTED_ENTITIES,
    ner=KoreanNER.from_pretrained(name_context_path="artifacts/name-context-v6-filter-retry"),
)
```

Requires optional NER dependencies and the explicitly downloaded pinned E5 backbone.
The repository Apache-2.0 license applies to the added classifier; underlying model
terms remain applicable. The default E5 configuration is unchanged.

Fresh synthetic evaluation: 192 sentences, 234 names, 48 non-person sentences.
TP/FP/FN changes from v3 225/8/9 to 225/0/9. All 225 previously correct individual
spans are preserved; no new false spans. Non-person false positives: 8/48 to 0/48.
Nine existing missed names remain. No claim of representative field accuracy.

See [evaluation, limits, rejection history and reproduction](../../docs/name-context-v6-evaluation.md).
The manifest freezes pre-training settings and hashes; the report records checkpoint
selection and the first fresh evaluation. Reproduce into a new output directory.
