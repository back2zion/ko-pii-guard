# Rejected v10 filter experiment

2026-10-10. This is not a loadable or accepted model.

The rescue head found all 702 development names, but none of 240 filter epochs
both preserved every train/validation name and rejected every known false proposal.
At best one required false proposal remained. Fresh evaluation was never run.

The frozen original source is `rejected_training_source.py`; its digest matches
`training_manifest.json`. See `rejected_training_report.json` for all selection results.
A subsequent experiment increases the loss weight on actual false proposals;
no threshold or person-name exception was added.
