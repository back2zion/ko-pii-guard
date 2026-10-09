# Changelog

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
