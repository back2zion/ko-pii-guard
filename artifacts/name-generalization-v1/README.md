# General character name experiment v1

Frozen E5 contextual features and person-label priors feed a character BIOES decoder.
Training mixes official KLUE training sentences with existing synthetic development
material. No lexical exception list or deterministic suffix stripping is introduced.

See [protocol and measured decision](../../docs/name-generalization-v1.md),
[data manifest](data-manifest.json), [training manifest](manifest.json),
[training report](report.json), and [frozen selection](selection.json).

The archived `*.source.txt` files match the source hashes used during training.
After that completed run, the current trainer's cache key was strengthened to include
the character-feature helper hash; this does not change the trained weights.
The original feature cache cannot be silently reused by the hardened trainer.

This artifact does not change the default runtime. Deployment status is determined
by the documented external evaluations and preservation gates, not training scores.
Raw KLUE/KDPII source text remains outside this repository.
