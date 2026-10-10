# Rejected v10 candidate

Known v9 misses (5) and false spans (2) were fixed, and 1,389 integration tests passed.
Historical 2,857-row acceptance protected all 3,005 correct v9 spans with no new false spans.

The first untouched v10 evaluation nevertheless lost the previously correct 공감 span
in `v10-evaluation-person-02-012`. Despite better aggregate TP/FP/FN
217/22/17 → 218/21/16, the candidate was rejected. No second checkpoint was selected
using this evaluation. The entire inspected v10 data was retired into v11 development.
See `training_report.json` for the first evaluation and per-span rejection.
