# Contextual name model v11 — explicit opt-in

2026-10-10. Synthetic development, validation and evaluation only.

```python
from ko_pii_guard.ner import KoreanNER
ner = KoreanNER.from_pretrained(name_context_path="artifacts/name-context-v11")
```

Fixes all five known v9 missed names and two false spans: the original 192-row
split now scores TP234/FP0/FN0. All 1,391 integration tests pass. Historical
3,049-row acceptance preserves all 3,222 v9-correct name/address spans and
introduces no false span.

First untouched v11 evaluation: TP/FP/FN 211/19/23 → 211/17/23; all 211 correct
spans retained, no new false span. Those FN23/FP17 remain. These results do not
establish accuracy on arbitrary real-world text or change the default model.

All v9 primary/rescue weights and the added v10 rescue are preserved. The learned
context filter expands to 128 hidden units, replays hard examples and requires
all train/validation gold below 0.5 non-person probability before selection.
Inference thresholds remain 0.99. Filter epoch 65 was selected before fresh eval.
No word/name exception or suffix-stripping rule is used.

The v10 candidate was rejected for losing the correct 공감 span on fresh data.
That data is retired into v11 development; this model uses a different fresh split.
A development-only v11 filter was also rejected before fresh evaluation.

Backbone: `FrameByFrame/korean-pii-e5-base`, revision
`a308c54b4407819624a5661e31e162a269f39818`; local/offline loading by default.
Parent hashes, frozen feature-cache hash and all selection results are recorded
in `training_manifest.json` and `training_report.json`. `training_source.txt`
preserves the exact executed source. Single-window filtering/rescue scope and
long-input fallback follow the existing runtime contract.

See [results, limitations and reproduction](../../docs/name-context-v11-evaluation.md).
