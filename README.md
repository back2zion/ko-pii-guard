# ko-pii-guard

Detect and mask Korean personal information in one line, built on [Microsoft Presidio](https://github.com/microsoft/presidio).

```python
from ko_pii_guard import KoreanPIIGuard

guard = KoreanPIIGuard()
guard.mask("홍길동 고객(주민번호 900101-1234567, 연락처 010-0000-0000)의 서류입니다")
# '홍길동 고객(주민번호 <KR_RRN>, 연락처 <PHONE_NUMBER>)의 서류입니다'
```

[한국어 안내](#한국어-안내)

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
- It adds two rules that cut common false positives in Korean business text.

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

Not covered: personal names, addresses, bank account numbers (formats vary by bank), and health insurance numbers. Names and addresses need a Korean NER model.

## Accuracy

Results on **synthetic** sentences from `benchmarks/synthetic_benchmark.py`, 200 samples per row. Identifiers are randomly generated with valid checksums and belong to no real person.

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

False positives: 0 out of 1,200 PII-free sentences (amounts, dates, order numbers, version strings, tracking numbers).

**Read these numbers with care.** The two false-positive rules and the tie-break were developed against this same synthetic set, so real-world accuracy will be lower. Issues and PRs with realistic (anonymized) failure cases are very welcome.

## Design notes

- **No NLP model.** Korean context matching looks only at the 20 characters before each match, not after, and uses no lemmas. This avoids a morphological-analyzer dependency.
- **Partial masking.** `style="partial"` keeps the birth-date part of RRN and FRN numbers (6 digits), the first 3 digits of phone numbers, and the first 2 characters of everything else.
- **Safety first for RRNs.** RRNs issued after October 2020 have no checksum, so any `YYMMDD-[1-4]XXXXXX` string is flagged at score 0.5. Set `score_threshold=0.6` to require a valid checksum or Korean context.
- **Tie-break.** When spans overlap with equal scores, Korean identifiers win over generic ones. For example, a 13-digit RRN can also pass the Luhn check for a card number.
- **Known gaps.** Identifiers glued together with no separator (for example `010-1234-5678010-2345-6789`) can be partly missed or mislabeled. Analysis time grows faster than input size on texts with thousands of matches.
- **Offline use.** Presidio's email recognizer uses `tldextract`, which tries to download the public suffix list once. If it can't, it logs a warning and falls back to its bundled snapshot.

## Disclaimer

This library reduces the risk of leaking personal information. It does not guarantee compliance with Korea's Personal Information Protection Act (개인정보 보호법) or any other regulation.

## Development

```bash
pip install -e ".[dev,guardrails]"
pytest -q
ruff check .
python benchmarks/synthetic_benchmark.py
```

## License

Apache-2.0. Built on [Microsoft Presidio](https://github.com/microsoft/presidio) (MIT).

---

## 한국어 안내

LLM 서비스의 입력과 출력, 로그에서 주민등록번호, 외국인등록번호, 사업자등록번호, 운전면허번호, 여권번호, 전화번호, 이메일, 카드번호를 찾아 마스킹합니다.

Microsoft Presidio에는 한국 식별번호 탐지기가 이미 들어 있습니다. 하지만 기본으로 꺼져 있고, 한국어 처리 설정이 따로 필요하며, 일부 `010` 번호를 놓치고, 한국어 문맥 단어(`주민번호`, `연락처` 등)를 인식하지 못합니다. 이 패키지는 이런 설정을 한 줄로 끝내 줍니다.

```python
from ko_pii_guard import KoreanPIIGuard

guard = KoreanPIIGuard()
guard.mask("주민번호 900101-1234567, 연락처 010-0000-0000")
# '주민번호 <KR_RRN>, 연락처 <PHONE_NUMBER>'
```

- 이름, 주소, 계좌번호, 건강보험증 번호는 아직 탐지하지 않습니다.
- 정확도 표는 합성 데이터 기준이며, 규칙을 같은 데이터로 다듬었기 때문에 실제 환경에서는 더 낮을 수 있습니다.
- 개인정보 보호법 준수를 보장하지 않습니다. 유출 위험을 줄이는 보조 도구로 사용하세요.
