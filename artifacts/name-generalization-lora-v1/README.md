# Actual-text name encoder adaptation v1

Research checkpoint; it is not selected automatically by `KoreanNER.from_pretrained()`.
Training and final external evaluation are separate stages. See the
[protocol and results](../../docs/name-generalization-v1.md) for current status.

The fixed E5 backbone is `FrameByFrame/korean-pii-e5-base` at revision
`a308c54b4407819624a5661e31e162a269f39818`. Rank-8, alpha-16 adapters update only
the attention query/value layers in its final four encoder blocks. The character
head starts from [the frozen-encoder experiment](../name-generalization-v1/README.md).
The base weights remain frozen. Original name priors and address predictions
come from a separate pass with the adapters disabled.

Training uses 28,909 rows, with 2,100 real-text validation rows, seed 20261012,
three epochs, and batch size 32. The checkpoint selection order is real validation
exact F1, fewer false positives, higher threshold, then earlier epoch. The previous
external 1,000 sentences are development data for posterior-decoder selection;
they are excluded from training. Fresh KLUE evaluation and KDPII are excluded from
all checkpoint and threshold selection.

Training completed with unchanged inputs and sources. Epoch 2 / threshold 0.7
was selected: validation TP/FP/FN 1468/44/30, full coverage 1474/1498, exact F1
97.54%. These validation figures do not establish external accuracy. Development
posterior selection is recorded separately from this training threshold.

The frozen final policy used posterior coverage cutoff 0.01. It was rejected:
fresh KLUE exact F1 improved from 84.58% to 89.14%, but 12 previously masked
names became exposed. KDPII false spans increased from 2 to 116. See the linked
results for both annotation policies, uncertainty, and the failed adoption gate.

`manifest.json` pins inputs, sources and training settings. `report.json` records
the selected epoch and validation metrics. `head.safetensors`
and `lora.safetensors` contain only the trained head and adapter weights. Exact
executed training sources are archived as `.source.txt`; `data-manifest.json` and
`training-text-hashes.json` record provenance and exact overlap checks.

The evaluation adapter performs two encoder passes per request. CUDA adapted
features use BF16 autocast; original priors use FP32. CPU uses FP32. Device
equivalence and latency must not be inferred from CUDA accuracy results.
Posterior coverage is a model path probability, not a calibrated privacy guarantee.

Source code is covered by the repository's Apache-2.0 license. The pinned upstream
E5 model declares MIT. KLUE data is CC-BY-SA-4.0, attributed to Park et al.,
*KLUE: Korean Language Understanding Evaluation* (NeurIPS 2021). Raw training
sentences and the large feature cache are kept outside this repository. These
source declarations do not establish the upstream model's complete data lineage.
