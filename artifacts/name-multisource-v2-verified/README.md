# Multisource name head v2 research checkpoint

Frozen E5 encoder and character BIOES head initialized from name-generalization-v1. This artifact records a bounded training experiment; it is not the default public runtime.

Training used 21,850 sentences: 9,850 original KDPII v1 train, 10,000 KLUE train, and 2,000 synthetic training sentences. Original offsets and all KDPII annotations are preserved. PS_NAME is the primary name label; PS_NICKNAME and PS_ID character positions are ignored in the training loss, not treated as ordinary negative labels.

The fixed budget was seed 20261013, 3 epochs, batch 64, CPU with 2 threads, AdamW learning rate 0.0005 and weight decay 0.01. All five BIOES/O classes have loss weight 1. The original head, vocabulary, and trainable prior_scale were retained. No particle/surface exceptions were introduced. 14,196 verified frozen feature rows were reused; 14,825 KDPII rows were encoded locally. Total elapsed time was 749.3 seconds.

Selection used equal macro exact F1 over KDPII official valid, NSMC and Wikitree validation, followed by fewer FP, higher threshold and earlier epoch. Thresholds were fixed at 0.5/0.7/0.9/0.95. Selected epoch **1**, threshold **0.9**, macro validation F1 **0.935488**.

| Source | TP | FP | FN | Fully covered | Exact F1 |
|---|---:|---:|---:|---:|---:|
| kdpii_v1_valid | 156 | 23 | 23 | 159 | 0.871508 |
| nsmc | 552 | 25 | 25 | 556 | 0.956672 |
| wikitree | 901 | 20 | 20 | 901 | 0.978284 |

KDPII validation retained 4,975 sentences after exclusions/deduplication (179 primary gold names). KLUE validation has 2,100 sentences. Synthetic validation has 96 sentences and does not select the checkpoint; its selected-threshold result was TP97/FP0/FN11, exact F1 0.946341. These are model-selection measurements, not independent final-test results.

The original encoder publisher used KLUE and KDPII; upstream membership has not been independently audited. The selected training/validation examples are new-head splits, not proven encoder-unseen examples. Sentence-level deduplication does not establish document/dialogue independence.

The legacy feature cache has only encoder, encoding_source, files and vocabulary metadata. Its archived original trainer SHA256 is pinned by the original training manifest. The current trainer differs only in main(); all remaining AST nodes match the archived source. The original manifest, archived character module and current character module hashes also match. Every cached source row, character ID and gold target was validated before reuse. Failed compatibility attempts are preserved separately and did not perform training.

training-text-hashes.json contains 45,930 exact UTF-8 SHA256 values: the complete initial checkpoint train/validation ancestry plus current train/validation. All source files remained unchanged throughout training. Initial checkpoint provenance and exact executed sources are archived alongside this file. Base weights are not duplicated here.

Sources: [KDPII v1](https://zenodo.org/records/10968609), CC-BY-4.0; [KLUE](https://github.com/KLUE-benchmark/KLUE/tree/3efd98708a40ff49251fddde35453f8fbb11f536), CC-BY-SA-4.0; project synthetic source, Apache-2.0. The pinned [E5 base model](https://huggingface.co/FrameByFrame/korean-pii-e5-base/tree/a308c54b4407819624a5661e31e162a269f39818) is tagged MIT. Dataset provenance and attribution are in data-manifest.json.

Subsequent development collection and strict public-runtime adoption gates are separate. No fresh reserved test annotations were opened for this training run.
