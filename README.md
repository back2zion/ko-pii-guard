# ko-pii-guard

Detect and mask Korean personal information in one line, built on [Microsoft Presidio](https://github.com/microsoft/presidio).

```python
from ko_pii_guard import KoreanPIIGuard

guard = KoreanPIIGuard()
guard.mask("홍길동 고객(주민번호 900101-1234567, 연락처 010-0000-0000)의 서류입니다")
# '홍길동 고객(주민번호 <KR_RRN>, 연락처 <PHONE_NUMBER>)의 서류입니다'
```

[한국어 안내](#한국어-안내)

This checkout includes **unreleased improvements** beyond PyPI v0.2.0. Use the
[development installation](#development) to try them. See [evaluation results and
reproduction commands](docs/evaluation.md) and [account-format sources](docs/account-format-sources.md).

## Why

Presidio already includes recognizers for Korean identifiers (resident registration, foreigner registration, business registration, driver's license, and passport numbers). Using them for Korean text still takes several non-obvious steps:

- The Korean recognizers are **disabled by default**.
- They are registered for language `ko`, which needs a separately configured NLP engine.
- The phone recognizer's default regions **do not include KR**. Even with KR enabled, `python-phonenumbers` rejects some `010-XXXX` ranges as invalid, so some real-format mobile numbers slip through.
- Context-word boosting needs tokens from an NLP engine. Without a Korean model, Korean context words such as `주민번호` or `여권번호` never raise confidence. The RRN and phone recognizers also have English-only context lists.

`ko-pii-guard` wires these together:

- It enables the Korean recognizers with a lightweight no-op NLP engine. No spaCy model is needed.
- It adds a Korean mobile pattern.
- It adds a Korean context-word booster.
- It checks Korean labels, sentence boundaries, identifier conflicts, and account layouts.
- It handles Korean particles after emails/cards and Unicode variations while returning original offsets.

## Install

```bash
pip install ko-pii-guard
pip install "ko-pii-guard[guardrails]"   # optional Guardrails AI validator
```

Requires Python 3.10+.

## Usage

```python
from ko_pii_guard import KoreanPIIGuard

guard = KoreanPIIGuard()  # score_threshold=0.4

guard.analyze("사업자등록번호 123-45-67891 입니다")
# [Finding(entity='KR_BRN', start=8, end=20, score=1.0, text='123-45-67891')]

guard.contains_pii("오늘 매출은 1,234,567원입니다")
# False

guard.mask("주민번호 900101-1234567", style="tag")  # '주민번호 <KR_RRN>'
guard.mask("주민번호 900101-1234567", style="partial")  # '주민번호 900101-*******'
guard.mask("연락처 010-0000-0000", style="partial")  # '연락처 010-****-****'
guard.mask("연락처 010-0000-0000", style="stars")  # '연락처 *************'

KoreanPIIGuard(entities=["PHONE_NUMBER", "EMAIL_ADDRESS"])  # detect only some types
```

### Guardrails AI

```python
from guardrails import Guard
from ko_pii_guard.guardrails_validator import KoreanPII

guard = Guard().use(KoreanPII(on_fail="fix"))
guard.validate("주민번호 900101-1234567 확인 부탁드립니다").validated_output
# '주민번호 <KR_RRN> 확인 부탁드립니다'
```

Guardrails AI sends usage telemetry unless it is disabled. To turn it off, run `guardrails configure --disable-metrics` or put `enable_metrics=false` in `~/.guardrailsrc`. `ko-pii-guard` adds no network calls of its own (see the offline-use note below).

## Supported entities

| Entity | What | Source |
|---|---|---|
| `KR_RRN` | Resident registration number (주민등록번호) | Presidio, checksum for pre-2020 numbers |
| `KR_FRN` | Foreigner registration number (외국인등록번호) | Presidio |
| `KR_BRN` | Business registration number (사업자등록번호) | Presidio, checksum |
| `KR_DRIVER_LICENSE` | Driver's license number (운전면허번호) | Presidio. Bare 12-digit runs need Korean context (this package) |
| `KR_PASSPORT` | Passport number (여권번호) | Presidio. Needs Korean context such as `여권번호` |
| `PHONE_NUMBER` | Korean phone numbers | Presidio (`KR` region) plus a Korean mobile pattern (this package) |
| `EMAIL_ADDRESS` | Email | Presidio |
| `CREDIT_CARD` | Card number | Presidio, Luhn check |
| `KR_ACCOUNT` | Korean bank account number (계좌번호) | 10–14 digits; generic 3-group candidates plus documented 4-group layouts. Bare and 4-group values require preceding Korean context |
| `KR_NAME` (experimental, opt-in) | Personal name (이름) | Explicit person fields; optional local NER for names inside sentences |
| `KR_ADDRESS` (experimental, opt-in) | Korean street/lot address (주소) | Administrative hierarchy or an address field, street/lot number and bounded unit details; optional local NER |

Health insurance numbers are not covered. Account and address detection recognizes
candidates; it does not verify bank ownership, account validity or address existence.

### Experimental names and addresses in sentences

The default profile keeps the nine identifier types above. Names and addresses
are under development and require explicit opt-in, including when a NER backend
is supplied. Their current evaluations are development regressions, not evidence
of production accuracy:

```python
from ko_pii_guard import DEFAULT_ENTITIES, KoreanPIIGuard

guard = KoreanPIIGuard(entities=[*DEFAULT_ENTITIES, "KR_NAME", "KR_ADDRESS"])
guard.mask("성명: 이상")  # '성명: <KR_NAME>'
```

This rule profile recognizes explicit fields, including rare surnames, compound
surnames and Unicode names. It also recognizes complete Korean road/lot addresses.
A bare place name such as `서울` is not a full address.

For general prose, enable the optional **local contextual NER** backend. Name length
or a surname list cannot distinguish `이상 없습니다` from `이상 씨가 왔습니다`.
This feature is currently unreleased; install from this checkout:

```bash
uv sync --frozen --extra dev --extra ner
uv run python -m ko_pii_guard.ner --download
```

```python
from ko_pii_guard import SUPPORTED_ENTITIES, KoreanPIIGuard
from ko_pii_guard.ner import KoreanNER

guard = KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=KoreanNER.from_pretrained())
guard.mask("이상 없습니다.")  # '이상 없습니다.'
guard.mask("이상 씨가 서류를 제출했습니다.")  # '<KR_NAME> 씨가 서류를 제출했습니다.'
guard.mask("소망 씨가 내일 방문합니다.")  # '<KR_NAME> 씨가 내일 방문합니다.'
guard.mask("새해 소망을 적어 주세요.")  # '새해 소망을 적어 주세요.'
```

Setup explicitly downloads pinned model weights (about 555 MB); model loading is
offline by default and document text stays local. This checkout uses CPU PyTorch.
NER is more expensive than the default rules: create and reuse the model once.
Long inputs use overlapping token windows. Model confidence is not a calibrated
probability. See [sources and model license declarations](docs/name-address-sources.md),
[name span annotation contract](docs/name-span-contract.md), and
[technical regression evaluation](docs/name-address-evaluation.md) for scope and limits.
The default model is `FrameByFrame/korean-pii-e5-base` at a fixed revision. We do
not apply its model card's suffix-removal heuristic: characters such as `은` can
belong to the name itself. Explicit non-person fields and missing-value markers
veto model predictions; address predictions need a numeric component. Filename
context does not suppress names: `김철수_이력서.pdf` contains a detectable name.

Detection and replacement policy are separate. `analyze` always reports accepted
findings; `mask(..., should_mask=callback)` lets the caller preserve selected spans,
for example a filename whose character range the application already knows:

```python
text = "첨부: 김철수_이력서.pdf, 연락처 010-1234-5678"
filename_start, filename_end = 4, text.index(",")
guard.mask(text, should_mask=lambda f: not (filename_start <= f.start < filename_end))
# '첨부: 김철수_이력서.pdf, 연락처 <PHONE_NUMBER>'
```

Returning `False` deliberately leaves that finding visible; it does not classify
the filename as non-PII. With no callback, all detected spans are masked.

## Accuracy — default identifier profile

Results measured on 2026-10-10 for the **unreleased checkout** on synthetic sentences from `benchmarks/synthetic_benchmark.py`, seed 2026, 200 samples per case, with the default threshold 0.4. A hit requires both the entity and the exact span to match. Checksums are generated where applicable. Account values use representative bank layouts, are unverified random test data, and are not guaranteed to be unassigned. The NH generator now uses the documented 4-group format; this changes the generated corpus from v0.2.0.

| Entity | Case | Recall |
|---|---|---|
| KR_RRN | `주민번호 … 로 본인확인 부탁드립니다` | 100% |
| KR_RRN | `번호는 … 입니다` (no dash) | 100% |
| KR_FRN | `외국인등록번호 … 확인` | 100% |
| KR_FRN | `등록번호 …` | 100% |
| KR_BRN | `사업자등록번호 …` / `거래처 번호 …` (no dash) | 100% / 100% |
| PHONE_NUMBER | mobile with and without dashes, Seoul landline | 100% |
| KR_DRIVER_LICENSE | with and without context | 100% |
| KR_PASSPORT | `여권번호 …` | 100% |
| KR_PASSPORT | `코드 …` (no context) | 0%, by design |
| EMAIL_ADDRESS, CREDIT_CARD | with context | 100% |

Account recall below is measured on synthetic data, 200 samples per bank and format (seed 2026, threshold 0.4), using `은행명 계좌번호 …` with dashes and `은행명 입금 계좌 …` without dashes. Each cell reports detected/200 and exact-span recall.

| KR_ACCOUNT layout | With dashes | Without dashes |
|---|---|---|
| 신한 | 200/200 (100.0%) | 200/200 (100.0%) |
| 국민 | 199/200 (99.5%) | 192/200 (96.0%) |
| 우리 | 184/200 (92.0%) | 154/200 (77.0%) |
| 하나 | 200/200 (100.0%) | 198/200 (99.0%) |
| 농협 | 200/200 (100.0%) | 177/200 (88.5%) |
| 기업 | 193/200 (96.5%) | 197/200 (98.5%) |
| 카카오뱅크 | 200/200 (100.0%) | 185/200 (92.5%) |
| 토스뱅크 | 200/200 (100.0%) | 200/200 (100.0%) |

The no-context account case (`코드 …`, undelimited) detects 0/200 as KR_ACCOUNT, by design. Some account candidates match existing identifier recognizers; these are deliberately classified as those identifiers rather than KR_ACCOUNT, reducing account recall. Representative layouts are not an exhaustive catalog of each bank's formats.

False positives: 0 flagged sentences and 0 findings out of 2,200 PII-free sentences (amounts, dates, order numbers, version strings, tracking numbers, account-like references, and product codes).

A separate collision probe (seed 20261009) prepends `은행 계좌 확인:` to synthetic RRN, BRN, mobile, and card numbers, with and without separators: 0/1,600 were mislabeled KR_ACCOUNT. These are positive PII cases, not part of the PII-free denominator.

**Read these numbers with care.** Rules were developed against this same synthetic set. This is a regression benchmark, not an independent real-world accuracy estimate. Issues and PRs with realistic (anonymized) failure cases are very welcome.

A separate **external synthetic slice** from K-PII-Bench contains 500 documents and 831 annotated spans across seven supported types (published test rows 10,000–10,499, zero-based; fixed revision and hashes in the [reports](benchmarks/results)). These documents were reserved until the detection changes were complete. Exact-span precision / recall / F1 changed from **97.67% / 65.70% / 78.56%** in v0.2.0 to **99.58% / 84.96% / 91.69%**. This is not a real-world false-positive rate, nor a comparison against other libraries. The authors' data includes format/checksum mismatches; all original labels remain in the denominator. See [methodology and limitations](docs/evaluation.md).

## Experimental name/address evaluation

Name and address detection remains under development. Small synthetic regression
sets help diagnose missed names, ordinary-word false positives and incorrect span
boundaries; they do not establish real-world accuracy. Current model selection
and thresholds used these development cases.

The [technical regression report](docs/name-address-evaluation.md) records results
and reproduction details. The [name span contract](docs/name-span-contract.md)
defines what counts as a name, while the
[context evaluation protocol](docs/name-context-evaluation.md) freezes the challenge
set before baseline evaluation and controlled scoring experiments. See the
[research sources](docs/name-address-sources.md) for methods considered and the
distinction between reading a paper and implementing its method.

## Design notes

- **Default rules and optional NER.** Numeric context matching looks at the 20 characters before each match. Names and addresses require explicit opt-in. Without a model, name detection requires explicit fields; optional NER uses sentence context. Neither a common surname nor two/three Korean syllables alone establishes a name.
- **Korean boundaries.** Labels can touch numbers (`계좌번호3333…`); trailing Korean particles stay outside email/card spans. ASCII identifier fragments remain excluded.
- **Unicode and offsets.** Fullwidth ASCII, Unicode decimal digits, common dashes and non-breaking spaces are folded for analysis. Selected invisible characters inside identifiers are removed. Findings and masks refer to the original Python string offsets. Set `normalize_unicode=False` for literal input analysis; this is not a general OCR correction system.
- **Account false positives.** Undelimited accounts require a preceding account keyword in the same clause. A nearer explicit label such as `주문번호` or `카드번호` rejects an account candidate. Date-shaped strings and fragments of longer identifiers are excluded. Existing identifier detections at score 0.4 or higher veto overlapping accounts, including when only KR_ACCOUNT is requested. Ambiguous values favor existing identifiers; account-like hyphenated codes without explicit labels can still be false positives.
- **Partial masking.** `style="partial"` keeps the birth-date part of RRN and FRN numbers (6 digits), the first 3 digits of phone and account numbers, the first character of names, no letters/digits of addresses, and the first 2 characters of other types.
- **Masking output.** Each finding gets its own replacement; separators between adjacent findings are preserved. Stars cover the complete span, including spans longer than 1,000 characters. This intentionally changes the upstream anonymizer's merging of adjacent same-type findings.
- **Safety first for RRNs.** RRNs issued after October 2020 have no checksum, so any `YYMMDD-[1-4]XXXXXX` string is flagged at score 0.5. Set `score_threshold=0.6` to require a valid checksum or Korean context.
- **Tie-break.** When spans overlap with equal scores, Korean identifiers win over generic ones. For example, a 13-digit RRN can also pass the Luhn check for a card number.
- **Calendar checks.** Unlabeled RRN/FRN candidates with impossible birth dates are rejected. An explicit Korean identifier label keeps a mistyped value eligible for masking. No new checksum requirement is imposed on post-2020 identifiers.
- **Known gaps.** Identifiers glued together with no separator can be partly missed or mislabeled. A confidence score is a heuristic, not a calibrated probability. A regex timeout raises an error rather than returning partially screened input.
- **Offline use.** Email validation uses the bundled public-suffix snapshot with network fetching and disk cache disabled. Update dependencies to refresh the snapshot. The core detector makes no network calls; optional Guardrails telemetry is separate.
- **Performance.** Project-owned deduplication and overlap selection use O(n log n) algorithms. Regex matching and phone parsing have their own costs, so this is not a whole-pipeline complexity guarantee. See [measured API timings](docs/evaluation.md).

## Disclaimer

This library reduces the risk of leaking personal information. It does not guarantee compliance with Korea's Personal Information Protection Act (개인정보 보호법) or any other regulation.

## Development

```bash
uv venv
uv sync --frozen --extra dev --extra guardrails
uv run pytest -q
uv run ruff check .
uv run python benchmarks/synthetic_benchmark.py --check
uv run python benchmarks/robustness_benchmark.py --check
uv run python benchmarks/name_address_benchmark.py --check
```

Keep `uv.lock` under version control. CI uses the same locked dependencies. For test runs with Guardrails telemetry disabled through OpenTelemetry, use `OTEL_SDK_DISABLED=true uv run pytest -q`.

Further reading: [CAPID (2026)](https://arxiv.org/abs/2602.10074) and [SPY (NAACL 2025)](https://aclanthology.org/2025.naacl-srw.23/). Their methods are not implemented in this package, and their results do not establish Korean bank-account accuracy.

For Korean usage and contribution guidance, see [개발·평가 안내](docs/development.md).

## License

Apache-2.0. Built on [Microsoft Presidio](https://github.com/microsoft/presidio) (MIT).

---

## 한국어 안내

LLM 서비스의 입력과 출력, 로그에서 주민등록번호, 외국인등록번호, 사업자등록번호, 운전면허번호, 여권번호, 전화번호, 이메일, 카드번호, 계좌번호를 찾아 마스킹합니다. 이름·주소 탐지는 별도로 켜는 실험적 기능으로 개발 중입니다.

Microsoft Presidio에는 한국 식별번호 탐지기가 이미 들어 있습니다. 하지만 기본으로 꺼져 있고, 한국어 처리 설정이 따로 필요하며, 일부 `010` 번호를 놓치고, 한국어 문맥 단어(`주민번호`, `연락처` 등)를 인식하지 못합니다. 이 패키지는 이런 설정을 한 줄로 끝내 줍니다.

```python
from ko_pii_guard import KoreanPIIGuard

guard = KoreanPIIGuard()
guard.mask("주민번호 900101-1234567, 연락처 010-0000-0000")
# '주민번호 <KR_RRN>, 연락처 <PHONE_NUMBER>'
```

- 이름·주소는 개발 중인 실험적 기능이며 명시적으로 켜야 합니다. 이름 규칙은 명시적인 이름 필드를 탐지합니다. 일반 문장에는 선택적인 로컬 NER 모델을 연결해야 합니다. `이상`, `소망`처럼 일반 단어와 같은 이름은 문맥으로 판단하며, 모델도 오탐·미탐과 조사 경계 오류가 있습니다. [주석 계약](docs/name-span-contract.md)과 [문맥 평가 계획](docs/name-context-evaluation.md)을 참고하세요.
- 주소는 도로명/지번·건물번호·일부 상세주소를 탐지하며 전국 주소의 존재 여부를 검증하지 않습니다. 이름·주소 기능은 현재 작업 트리에 포함된 미배포 기능입니다. 건강보험증 번호는 지원하지 않습니다.
- 계좌번호는 형식과 앞쪽 문맥으로 탐지하며 실제 계좌의 유효성을 검증하지 않습니다. 구분자 없는 숫자는 문맥이 필수이고, 기존 식별번호와 겹치면 기존 유형을 우선합니다.
- 식별번호 정확도 표는 합성 데이터 기준이며, 규칙을 같은 데이터로 다듬었기 때문에 실제 환경에서는 더 낮을 수 있습니다. 이름·주소의 소규모 개발 회귀 결과는 [기술 평가 문서](docs/name-address-evaluation.md)에 기록하며 일반적인 정확도로 제시하지 않습니다.
- 개인정보 보호법 준수를 보장하지 않습니다. 유출 위험을 줄이는 보조 도구로 사용하세요.
