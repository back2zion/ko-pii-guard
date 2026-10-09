`robustness.jsonl` is a manually curated **synthetic development regression set**.
It covers Korean labels adjacent to numbers, numeric token boundaries, misleading
context, collisions with existing identifier types, and Unicode formatting.
The annotations specify the expected entity and original Python string offsets
(`start` inclusive, `end` exclusive). Expected labels were assigned to the
constructed examples, not copied from model predictions.

These examples are also used in the test suite. Results therefore measure
regression behavior on known cases, **not held-out accuracy or real-world
precision**. The numbers are test examples with no verified ownership; they are
not guaranteed to be unassigned. Account examples do not validate bank ownership,
account existence, or a complete list of bank formats.

Run `uv run python benchmarks/robustness_benchmark.py --output /tmp/robustness.json`.
The report includes exact entity-and-span precision, recall and F1, false-positive
sentence counts, and the proportion of expected spans fully replaced by stars
in the actual masking output. It also records every error and the corpus SHA-256
so reported measurements can be tied to the exact input data.
