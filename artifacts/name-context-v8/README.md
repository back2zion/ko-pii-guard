# Rejected Korean name experiment v8

Do not adopt. It fixed the six known v7 missed names and four false spans and passed
1,378 historical tests, but the first fresh evaluation lost one previously correct
name span. The strict per-span gate rejected it despite better aggregate TP/FN.

The refined rescue forgot a span which the v7 rescue still found at 0.998169 confidence.
The updated non-person filter did not reject it. Subsequent work preserves the old
rescue and adds the refined decoder instead of replacing its weights.

See [evaluation, provenance and rejection](../../docs/name-context-v8-evaluation.md).
