# Changelog

## Unreleased

## 0.3.0 — 2026-10-11

Highlights: opt-in `KR_NAME` and `KR_ADDRESS`, an optional local NER (`[ner]` extra), the `should_mask` callback, Unicode normalization with original offsets, and account-format and span fixes. The default nine identifier entities are unchanged. Learned name-context heads and model artifacts live in the repository only; they are not shipped in the PyPI package.

Name and address detection is **opt-in**. `KR_NAME` and `KR_ADDRESS` are not enabled by default; the default set of nine identifier entities is unchanged. The experimental name-context heads (v2 through v11) are likewise opt-in and never the default; the default decoder remains E5. Per-version detail: [docs/name-context-history.md](docs/name-context-history.md). Requirements and quality contracts: [docs/PRD.md](docs/PRD.md).

### Added

- Opt-in `KR_NAME` and `KR_ADDRESS` entities; the default nine identifier entities are unchanged. Includes explicit name fields, bounded road/lot address grammar, reference parentheses and wrapped address lines.
- Optional local contextual name/address NER with pinned safetensors, offline loading by default and overlapping windows for long documents. A separate optional-NER CI job and a business Korean boilerplate corpus were added.
- Opt-in learned name-context heads, from an experimental character name decoder through v2, v3, a non-person candidate filter (byte-identical v3 name decoder weights), and rescue heads v7, v9 and v11. Each version preserves earlier decoder/filter weights and accepted spans. The latest, v11, adds preserved decoders, an expanded filter and hard-example replay; on its synthetic evaluation it retains all 211 correct spans and reduces FP 19 to 17 (FN 23 / FP 17 remain). Earlier versions preserved all baseline-correct spans (for example 225 for the filter, 228 for v9) with no new false span. See [docs/name-context-history.md](docs/name-context-history.md).
- Evaluation and CI infrastructure for name work: frozen, split-separated synthetic evaluations with checkpoint-selection provenance, named Red/Green exact-span and masking regression contracts, strict per-span non-regression CI gates, and a benchmark-only lexicon score ablation. Reports keep remaining misses and false positives. Full optional-model integration: 1,391 tests pass.
- Documented four-group account candidates with mandatory local context.
- Optional `should_mask` callback separating detection from replacement.
- Independent name/address fixture annotations, per-track exact-span and actual full-masking evaluation, including common-word/name contrasts.
- Source-backed format documentation, curated regression cases, external evaluation and reproducible performance reports; benchmark failure gates in CI.
- Product requirements document ([docs/PRD.md](docs/PRD.md)) with quality contracts and development priorities; documented the `KR_NAME` span contract and a recent Korean de-identification research review. Small name/address diagnostics are kept out of README accuracy claims.

### Changed

- Presidio validators are used with O(n log n) deduplication/overlap handling; masks are rendered directly.
- Spans longer than 1,000 characters are fully masked, and separators between adjacent findings are preserved.
- Partial masking preserves one name character and masks all address letters/digits.
- Identifier typography and interior invisible characters are normalized while preserving original offsets.
- RRN/FRN calendar dates are checked without losing explicitly labeled mistyped identifiers.
- The package version is derived from installed metadata.
- Public-suffix network fetching is disabled; regex timeouts propagate instead of silently skipping rules.
- Malformed custom NER result objects and nonnumeric/boolean confidence values are rejected explicitly.
- The explicit non-person owner field veto is applied consistently to contextual model predictions.
- The NH synthetic account layout and the name-field corpus's prose-name annotation were corrected; v1 and historical reports are preserved as development history.

### Fixed

- Name-context regressions closed as named Red/Green tests: 21 known v1 misses and seven regressions (v2), 26 v2 error sentences (v3), all nine known v6 misses (v7), six v7 misses and four false spans (v9; original evaluation TP234/FP0/FN0), and five v9 missed names and two false spans (v11).
- Complete middle-dot names are preserved in explicit person fields, and tab/multiple-space honorifics are separated; before/after boundary regression reports added.
- Names are still detected in filenames.
- Korean labels touching account numbers and Korean particles following emails/cards.
- Short numeric codes, ASCII token fragments, misleading bank words and cross-sentence context are no longer accepted as account candidates.

### Rejected / not adopted

- multisource-v2 `selective:addition=1.0:removal=0.9999` checkpoint: an earlier operational release decision was incorrect and has been retracted. The policy only strips baseline spans and never adds one; on the KDPII heldout it is identical to the original E5 baseline (F1 85.71%/71.28%) and KLUE WikiTree (418/418), and introduces one KLUE NSMC regression ("가르시아" incorrectly stripped). Net effect is no gain and one regression, so the automatic gate's `quality_gate_passed: false` was correct and the checkpoint is not promoted. The earlier "0.13 to 0.86" claim compared it against a different, already-rejected candidate rather than its own baseline. See [docs/name-multisource-v2.md](docs/name-multisource-v2.md#운영-배포-결정--철회-원래-승인은-틀린-비교에-근거함).
- v4 and v5 (recall regressions), v8 (lost one previously correct name) and v10 (a fresh evaluation lost a correct name) were rejected; rejected runs are documented in [docs/name-context-history.md](docs/name-context-history.md). The default E5 decoder was retained because regressions remained in the experimental character decoder.

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
