# Changelog

## Unreleased

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
