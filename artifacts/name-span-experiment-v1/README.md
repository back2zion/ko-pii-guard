# Joint name span experiment v1

Research artifacts only; not selected by the default runtime.

Three fixed seeds were trained for six epochs each. Selection used validation only,
before the external evaluation. See [protocol and final decision](../../docs/name-span-experiment-v1.md),
[training report](report.json), [public API evaluation](../../benchmarks/results/name-span-v1-public.json),
and [external evaluation](../../benchmarks/results/name-span-v1-external.json).

The public API diagnosis reduced TP/FP/FN from 211/17/23 to 232/2/2 for seed
20261010 on previously inspected synthetic material. Seed 20261012 introduced
a false span and failed that regression gate. This is not evidence of universal
accuracy or preservation of every historical corpus.

`manifest.json` fixes source/data hashes and conditions. Executed model/trainer
source snapshots and TDD failure transcripts are included. `validation.json`
records the existing full-suite result, not a full historical acceptance of these
new weights. Reproduction commands are in the protocol. External raw data stays
outside the repository; provenance and derived coordinate metrics are recorded.

Final decision: reject all three candidates for runtime adoption. External exact
span F1 was 59.79–64.73%, below v11 (73.75%) and the base E5 reference (85.05%).
All three failed external preservation gates. No threshold or seed was selected
using these external results. See the linked report for coordinate failures and
source provenance.
