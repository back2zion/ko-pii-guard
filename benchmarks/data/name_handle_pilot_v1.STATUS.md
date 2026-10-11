# name_handle_pilot_v1 - STATUS: PROVISIONAL (not established ground truth)

- `reviews.jsonl`, `adjudication_queue.jsonl`, `audit_sample.jsonl`, `review-summary.json` are **provisional** raw merges of two
  blind agent reviews. They are preserved unchanged and must not be used as labels.
- Reviewer A's explicit judgments cover 203 items; 197 items were filled by a script default (`default_filled`, `review_unverified`,
  see `reviewer_A_origin.json`). 190 of the 335 `agent_consensus` rows are default-filled non_personal rows and are NOT validated
  agreement; only 145 consensus rows (all with explicit A judgments) can count as agent agreement.
- The queue (65) and audit sample (34) are not the final workload: regenerate them after separating the unverified rows.
- 89.5% agreement and kappa 0.819 describe agreement between the raw outputs only, not correctness.
- No policy is confirmed (see `docs/name-handle-policy-proposal.md`). No row was adjudicated or audited by the project owner.
