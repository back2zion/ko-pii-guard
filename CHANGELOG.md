# Changelog

## Unreleased

- Complete the interrupted KDPII heldout evaluation for the multisource-v2 checkpoint (PS_NAME F1 0.86, PS_NAME+NICKNAME F1 0.71, both passing); the automatic release gate still fails on a single KLUE NSMC name regression ("가르시아"), so `quality_gate_passed` stays false and the checkpoint ships only via a documented human operational decision, not a gate change. See [name-multisource-v2.md](docs/name-multisource-v2.md#운영-배포-결정-게이트-통과가-아닌-사람의-승인).
- Fix five v9 missed names and two false spans with exact masking regressions. Reject v10 after a fresh evaluation loses a correct name; adopt opt-in v11 with preserved decoders, an expanded filter and hard-example replay. All 1,391 tests pass; 3,222 historical correct spans are preserved with no new false span. New synthetic evaluation retains all 211 correct spans and reduces FP19→17; FN23/FP17 remain. Add CI per-span gates and document the rejected runs.

- Fix the six known v7 missed names and four false spans as ten named Red/Green contracts; original evaluation now TP234/FP0/FN0 and all 1,381 integration tests pass.
- Reject v8 for losing one previously correct name; preserve the original v7 rescue and add the refined decoder in opt-in v9. Freeze a new evaluation and CI per-span gates: all 228 correct spans preserved, no new false span, TP/FP/FN 228/3/6 to 229/2/5. Document the five different remaining misses and two false spans.

- Close all nine known v6 misses with named Red/Green exact-span and masking tests. Add an opt-in v7 rescue head while preserving v6 decoder/filter weights and accepted spans; all 1,366 tests pass.
- Freeze a separate v7 evaluation and enforce per-span nonregression in CI: preserve 224 correct spans with no new false span; TP/FP/FN improves 224/4/10 to 228/4/6. Retain the remaining six new misses and four existing false spans in the report.

- Add an opt-in learned non-person candidate filter with byte-identical v3 name decoder weights; preserve all 225 baseline-correct spans on a new 192-sentence evaluation while reducing false-positive spans from 8 to 0 (9 existing misses remain).
- Reject v4/v5 recall regressions; add exact-span/masking contracts, strict per-span CI gates and frozen evaluation provenance. Full optional-model integration: 1,354 tests passed.

- Fix the remaining v2 error sentences as 26 named Red/Green regression tests; train an opt-in v3 head with all inspected v2 data explicitly retired into development, and evaluate a separately frozen 280-sentence synthetic split after checkpoint selection. Preserve remaining false positives and misses in the evaluation report.
- Train and validate opt-in name context head v2 with single-character, filename, multiple-person and common-word contrast examples; close the 21 known v1 misses and seven regressions without changing their gold annotations.
- Preserve v1 and inspected data as development history; add new split-separated v2 evaluation, explicit checkpoint selection provenance and CI acceptance gates.
- Add an experimental opt-in character name decoder, reproducible local training, a split-separated synthetic evaluation and a recent Korean de-identification research review; retain the default E5 decoder because regressions remain.
- Apply the explicit non-person owner field veto consistently to contextual model predictions.
- Correct the name-field corpus's prose-name annotation, preserving v1 and historical reports.
- Preserve complete middle-dot names in explicit person fields and separate tab/multiple-space honorifics; add before/after boundary regression reports.
- Reject malformed custom NER result objects and nonnumeric/boolean confidence values explicitly.
- Add a product requirements document with quality contracts and development priorities.
- Document the KR_NAME span contract and keep small name/address diagnostics out of README accuracy claims.
- Add a frozen synthetic name-context contrast evaluation and benchmark-only lexicon score ablation before considering retraining.
- Add opt-in KR_NAME and KR_ADDRESS; keep the default nine identifier entities unchanged.
- Add explicit name fields, bounded road/lot address grammar, reference parentheses and wrapped address lines.
- Separate detection from replacement with the optional should_mask callback; preserve name detection in filenames.
- Add a business Korean boilerplate corpus and a separate optional-NER CI job.
- Add optional local contextual name/address NER with pinned safetensors, offline loading by default and overlapping windows for long documents.
- Preserve one name character and mask all address letters/digits in partial masking.
- Add independent name/address fixture annotations, per-track exact-span and actual full-masking evaluation, including common-word/name contrasts.
- Add documented four-group account candidates with mandatory local context; correct the NH synthetic layout.
- Fix Korean labels touching account numbers and Korean particles following emails/cards.
- Reject short numeric codes, ASCII token fragments, misleading bank words and cross-sentence context.
- Normalize identifier typography and interior invisible characters while preserving original offsets.
- Check RRN/FRN calendar dates without losing explicitly labeled mistyped identifiers.
- Use Presidio validators with O(n log n) deduplication/overlap handling; render masks directly.
- Fully mask spans longer than 1,000 characters and preserve separators between adjacent findings.
- Disable public-suffix network fetching; propagate regex timeouts instead of silently skipping rules.
- Add source-backed format documentation, curated regression cases, external evaluation and reproducible performance reports.
- Add benchmark failure gates to CI and derive the package version from installed metadata.

## 0.2.0 — 2026-10-09

- Add KR_ACCOUNT for hyphenated bank accounts and context-required 10–14-digit runs.
- Keep existing identifiers ahead of account candidates, including account-only queries.
- Reject date-shaped candidates, identifier fragments, and conflicting nearby labels.
- Keep three account digits visible in partial masking.
- Add representative bank-layout generators, exact-span recall, and expanded negative cases.
- Migrate development, CI, and package builds to uv; commit uv.lock.

## 0.1.0 — 2026-10-08

- First release: Korean PII detection and masking on top of Microsoft Presidio.
- Entities: KR_RRN, KR_FRN, KR_BRN, KR_DRIVER_LICENSE, KR_PASSPORT, PHONE_NUMBER, EMAIL_ADDRESS, CREDIT_CARD.
- Korean context-word booster, Korean mobile pattern, false-positive rules for bare digit runs.
- Optional Guardrails AI validator (`back2zion/korean_pii`).
