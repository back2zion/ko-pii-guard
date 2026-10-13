# 평가와 재현

측정일: **2026-10-09~10**. 대상은 현재 **미배포 개발 버전**과 git 태그 `v0.2.0`입니다.
실환경 최고 성능이나 개인정보 누출 방지를 보장하는 시험은 아닙니다.

## 2026-10-13 독립 비교와 문맥 검사 최적화

기준 커밋 `0a85e451`과 현재 작업 트리에서 공개 K-PII-Bench의 별도 합성
500문서(20,000~20,499행)를 비교했다. 지원 7유형의 정확한 유형·구간은 두 버전
모두 TP 725, FP 8, FN 121(F1 91.83%)이었다. 외부 KDPII의 고정 500 대화문장에서는
지원 구조화 유형의 정답 39구간 중 두 버전 모두 TP 20, FP 6, FN 19(F1 61.54%)였다.
대화문과 합성 문서 어느 쪽도 실제 사내 문서 정확도의 대용은 아니다.

계좌 문맥 검색을 정규식 반복 대신 문자열 검색으로 바꾼 뒤 같은 합성 휴대전화
1,000개 문서의 `analyze()` 5회 중앙값은 134.9ms에서 104.4ms로 줄었다.
공개 합성 혼합 문서 500건·109,394자의 전체 탐지는 기준 196.8ms, 현재 185.8ms,
별표 마스킹은 195.6ms에서 186.6ms였다(각 10회 중앙값). 이 혼합 문서의 정확도
수치는 바뀌지 않았다. 작업·환경에 따른
차이가 있으므로 전화번호 반복 자료의 속도 차이를 일반 문서 전체에 적용하지 않는다.
[최적화 전 독립 정확도 비교](../benchmarks/results/current-vs-baseline-independent-2026-10-13.json),
[최적화 후 속도 원본](../benchmarks/results/context-scan-optimization-2026-10-13.json)을
참고한다. 실무 문서는 [비공개 로컬 평가 절차](private-evaluation.md)로 따로 검증한다.
별도 생성형 강건성 검사에서는 고정 seed의 5만 합성 문장에 대해 탐지 좌표, 별표
마스킹, `mask_with_details()`의 출력 좌표 불변식 오류가 없었다.
[불변식 원본](../benchmarks/results/masking-invariants-2026-10-13.json)은 탐지 정확도
점수가 아니다.

## 2026-10-13 계좌·카드 충돌 개선

`benchmarks/account_collision_benchmark.py`는 출처가 확인된 농협 `302-4-4-2`와
카카오뱅크 `3333-2-7` 구조의 구분자 없는 13자리 합성 값을 각 300개 생성한다.
형식의 출처와 적용 범위는 [계좌 형식 근거](account-format-sources.md)를 따른다.
각 값 앞에는 해당 은행명과 `입금 계좌`를 두고 정확한 유형·구간과 실제 별표 마스킹을 검사한다.
또 같은 값에 `카드번호` 라벨을 붙인 반례와 16자리 카드 300개의 계좌 오분류를 검사한다.

| 같은 고정 합성 자료 | 변경 전 `0a85e451` | 이번 작업 트리 |
|---|---:|---:|
| 농협 정확한 `KR_ACCOUNT` 유형·구간 | 264/300 | 300/300 |
| 카카오뱅크 정확한 `KR_ACCOUNT` 유형·구간 | 264/300 | 300/300 |
| 전체 계좌 값 별표 마스킹 | 600/600 | 600/600 |
| 계좌 라벨 아래 카드 오분류 | 72/600 | 0/600 |
| 카드 라벨 아래 우연히 카드 체크섬을 통과한 값의 카드 탐지 | 72/72 | 72/72 |
| 별도 16자리 카드의 계좌 오분류 | 0/300 | 0/300 |

계좌 생성 seed는 `20261014`, 16자리 카드 seed는 `20261015`이며 사례 입력의 SHA-256은
`2df4f5520c9c36a27c80f1ddb37c0545d9091e097ed8c5ffe6d5a555e1b2a93c`이다.
기준 코드는 위 커밋의 `src`만 임시 디렉터리에 추출하고 같은 Python 3.12.14 및 잠긴
의존성에서 실행했다. 아래 명령은 현재 소스 결과를 재현한다.

```bash
uv run --frozen python benchmarks/account_collision_benchmark.py --check
```

이 자료는 구현 후에 만든 **개발용 합성 회귀**다. 형식이 실제 발급 계좌라는 뜻이 아니며,
개발·선택에 사용한 분포이므로 일반적인 계좌 탐지 정확도나 독립 평가로 해석하지 않는다.
변경 전에도 600개 값은 카드번호 태그로 가려졌으므로 이 수치는 개인정보 가림 재현율의
증가가 아니라 **유형 분류의 개선**이다. 기존 전체 합성 음성 2,200개·식별번호 충돌
1,600개의 회귀 검사는 별도로 통과해야 한다.

같은 구조가 JSON의 `account_number`에 들어가면 기존 구현은 자매 필드의 은행명을
문맥에 쓰지 못해 카드로 분류했다. 현재는 동일 객체의 `bank`, `bank_name`, `은행`,
`은행명`이 명시적으로 농협 또는 카카오뱅크를 가리킬 때 해당 계좌 필드에 문맥을
전달한다. 일치하지 않는 은행이나 별도 객체의 은행은 계좌 우선 근거로 쓰지 않는다.
이 역시 유형 분류 회귀이며 실제 발급 계좌의 검증은 아니다.
같은 고정 합성 600계좌를 JSON 필드로 검사하면 현재 유형은 600/600 `KR_ACCOUNT`였고,
별도 16자리 카드 300개가 JSON 계좌로 오분류된 건수는 0이었다.

`card_number`처럼 명시적인 JSON 카드 필드는 체크섬 오타 때문에 일반 카드 인식기에서
빠져도, 값 전체가 [Visa의 PAN 길이 설명](https://developer.visa.com/capabilities/visa-secure-remote-commerce/glossary)에
맞는 13~19자리 숫자 형태라면 가린다. 공백·하이픈 한 글자 구분자만 허용하고 다른 키나
일반 문장으로 확장하지 않는다. 이는 공개 독립 정확도 상승 수치가 아닌 가림 정책 회귀다.

## 기본 프로필: 기존 식별번호 9종

`benchmarks/data/robustness.jsonl`은 사람이 만든 합성 사례 95개입니다. 한국어 붙여쓰기,
조사, 숫자 경계, 잘못 연결된 문맥, 기존 식별번호와의 충돌, 유니코드 표기를 검사합니다.
정답 구간 57개·음성 문장 39개이며, 구현과 테스트에 사용했으므로 독립 평가가 아닙니다.

| 지표 — 같은 95개 개발 사례 | v0.2.0 | 개발 버전 |
|---|---:|---:|
| 정확한 구간+유형 precision | 74.19% | 100% |
| 정확한 구간+유형 recall | 80.70% | 100% |
| F1 | 77.31% | 100% |
| 실제 stars 마스킹으로 전체가 가려진 정답 구간 | 46/57 | 57/57 |
| 음성 문장 오탐 | 16/39 | 0/39 |

별도의 기존 합성 생성 벤치마크는 사례당 200개, seed 2026입니다. 농협 생성기는 공식 자료의
4그룹 구조로 정정했으므로 v0.2.0 문서의 은행별 재현율과 직접 비교하지 않습니다.
원문에 없던 유형/추가 구간, 전체 마스킹 여부도 JSON에 기록합니다. 문맥 없는 여권·계좌
행은 의도적으로 탐지하지 않는 정책 확인 사례입니다.

```bash
uv run python benchmarks/robustness_benchmark.py --check --output /tmp/robustness.json
uv run python benchmarks/synthetic_benchmark.py --check --output /tmp/synthetic.json
```

원본: [회귀 기준선](../benchmarks/results/robustness-v0.2.0.json),
[회귀 개발 버전](../benchmarks/results/robustness-current.json),
[합성 생성 결과](../benchmarks/results/synthetic-current.json).

## 외부 한국어 데이터

데이터는 Park, Oh, Cha의 [K-PII-Bench](https://huggingface.co/datasets/woohyun212/k-pii-bench),
CC BY 4.0입니다. 고정 revision은 `844f92f7beba98fa964f957ba324017e82451701`입니다.
첫 500문서는 오류 진단에 사용했습니다. 이메일 뒤의 조사와 카드번호 경계 문제를 발견한
개발용 외부 표본이므로 이 첫 표본을 미사용 평가 데이터라고 주장하지 않습니다.

아래 수치는 구현을 마친 뒤 별도로 평가한 `test_track_a`의 **0-based 10,000~10,499행,
500문서, 12개 도메인, 지원하는 7유형의 정답 구간 831개** 기준입니다. 이 표본의 오류를
보고 탐지 규칙을 추가 조정하지 않았습니다. v0.2.0 소스도 같은 uv 환경에서 실행했습니다.

| 같은 외부 합성 500문서 | v0.2.0 | 개발 버전 |
|---|---:|---:|
| 정확한 구간+유형 TP / FP / FN | 546 / 13 / 285 | 706 / 3 / 125 |
| Precision | 97.67% | 99.58% |
| Recall | 65.70% | 84.96% |
| F1 | 78.56% | 91.69% |
| 유형과 무관한 전체 구간 탐지 범위 | 66.06% | 85.32% |

`KR_PHONE_NUMBER→PHONE_NUMBER`, `KR_BANK_ACCOUNT→KR_ACCOUNT`로 이름을 대응시켰습니다.
이름·주소는 이 기본 프로필의 평가 대상에서 제외하고 유형별 개수를 보고서에 기록합니다.
지원하는 유형의 모든 정답은 형식·체크섬 불일치가 있어도 미탐 분모에 그대로 남깁니다.
마지막 coverage 지표는 예측 구간 합집합이 정답 구간 전체를 덮는 비율이며 유형 정확성과 다릅니다.

[저자 datasheet](https://github.com/woohyun212/k-pii-bench/blob/main/DATASHEET.md)에 따르면
이 데이터는 합성이며 비PII 방해값이 별도로 주입되지 않았습니다. 따라서 위 precision을
운영 환경의 오탐률로 해석하면 안 됩니다. 공개판은 원 논문의 일부 라벨 오류와 분할을
수정했으므로 원 논문의 다른 모델 점수와 직접 비교하지 않습니다. 데이터셋 원문은 이 저장소에
재배포하지 않고 평가 시 내려받습니다. 보고서에는 오류 문서 ID와 위치만 저장합니다.

```bash
uv run python benchmarks/external_benchmark.py --offset 10000 --limit 500 \
  --cache /tmp/k-pii-heldout-500.jsonl --output /tmp/external.json --label current
```

캐시 파일은 해당 offset/limit 전용으로 사용해야 합니다. 다른 표본을 측정할 때는 새 경로를
지정하세요. 보고서의 SHA-256이 같은지 확인하면 동일 입력 여부를 검증할 수 있습니다.
원본: [기준선](../benchmarks/results/external-holdout-v0.2.0.json),
[개발 버전](../benchmarks/results/external-holdout-current.json).

## 선택 프로필: 개발 중인 이름·주소의 기술 회귀 결과

기본 `DEFAULT_ENTITIES`는 기존 9종이다. 이름·주소를 포함하려면
`KoreanPIIGuard(entities=SUPPORTED_ENTITIES, ner=KoreanNER.from_pretrained())`처럼
명시적으로 켠다. NER 없이 같은 entities를 선택하면 필드·주소 규칙만 동작한다.
아래 두 자료는 사람이 만든 **합성 개발 회귀 세트**이며 모델 선택·규칙 수정·임계값 선택에
사용했다. 별도의 실사용 독립 평가나 논문 방법의 재현 성능이 아니다.

아래 이름·주소 수치는 **작은 개발 회귀 세트의 진단 기록**이다. 특히 업무 문장 `0/62`는
해당 부류의 재발 방지 검사이고, 자유 문장 이름 `15/18`은 공개 재현율을 추정할 표본이 아니다.
README 성능표나 대외 소개에는 이 값을 싣지 않는다. 문장 수를 늘려도 같은 틀을 반복하면
독립적인 표본 수가 늘어난 것은 아니다. [추가 문맥 대조 평가](name-context-evaluation.md)도
이름·문장 틀·분할별 진단으로 보고하며, 기본 식별번호 지표와 합산하지 않는다.

기본 식별번호 프로필의 수치는 이름 모델과 합산하지 않는다:

| 기본 프로필 검사 | 실측 |
|---|---:|
| seed 2026, 사례당 200개의 숫자 중심 음성 문장 | 오탐 0/2,200 |
| 기존 주민·사업자·전화·카드의 계좌 오분류 | 0/1,600 |

**이름·주소 109문장, 정답 75구간(이름 47·주소 28), 음성 38문장**, 규칙0.4 / E5 NER0.9:

| 선택 프로필 | 엔티티 | Precision | Recall | F1 | 실제 전체 마스킹 |
|---|---|---:|---:|---:|---:|
| 규칙 | KR_NAME | 100% | 61.70% | 76.32% | 29/47 |
| 규칙+NER | KR_NAME | 100% | 93.62% | 96.70% | 44/47 |
| 규칙 | KR_ADDRESS | 100% | 96.43% | 98.18% | 27/28 |
| 규칙+NER | KR_ADDRESS | 100% | 100% | 100% | 28/28 |

두 프로필 모두 이 자료의 음성 오탐은 **0/38**이다. 구조화74문장과 자유문장35문장을
분리하면 NER 자유문장 이름은 **15/18(83.33%)**, 주소는 **6/6**을 정확히 탐지·완전히
가렸다. 이름3개는 여전히 놓친다. 0.9보다 낮은 점수의 복성 이름, 동료 문맥의 `소망`,
라틴 문자 이름을 분모에 남겼다. 높은 임계값이 모든 모호성을 해결하지는 않는다.
두 파일명 문장은 인물 참조 여부가 모호해 비채점 출력으로 공개한다.
[판정 정책 변경과 파일명 양성 사례](name-address-evaluation.md)를 함께 읽어야 한다.

**업무 상용구 별도 세트82문장(음성62, 양성20문장·이름21구간):**

| 선택 프로필 | 음성 오탐 | 이름 Precision | Recall | F1 | 실제 전체 마스킹 |
|---|---:|---:|---:|---:|---:|
| 규칙 | 0/62 | 100% | 57.14% | 72.73% | 12/21 |
| 규칙+NER | 0/62 | 100% | 95.24% | 97.56% | 20/21 |

NER 포함11종 프로필도 기본 프로필과 동일한 숫자 중심 음성 자료에서 **0/2,200**이었다.
이는 이름의 중의성 검증을 대신하지 않는다. 작은 개발 세트의 0건을 실서비스 오탐0으로
확대하지 않는다. 테스트 분모를 합쳐 하나의 더 큰 정확도 수치로 발표하지 않는다.

```bash
uv sync --frozen --extra dev --extra guardrails --extra ner
uv run python -m ko_pii_guard.ner --download
KO_PII_TEST_NER=1 uv run pytest -q tests/test_ner.py
uv run python benchmarks/name_address_benchmark.py --ner --numeric-negatives --check --check-negatives --output /tmp/name-address.json
uv run python benchmarks/name_address_benchmark.py --data benchmarks/data/business_korean.jsonl --ner --check-negatives --output /tmp/business-korean.json
```

모델 고정 revision·임계값·데이터/소스 SHA·오류ID는
[이름·주소 원본](../benchmarks/results/name-address.json),
[업무 상용구 원본](../benchmarks/results/business-korean.json)에 있다.
[학습 자료와 모델 출처](name-address-sources.md),
[업무 세트의 판정 기준](business-korean-evaluation.md)도 별도로 기록했다.
기본 CI와 별도 NER CI가 음성 회귀를 검사한다. 이름·주소 CI는 외부API에 문서를 보내지
않고 고정 모델을 다운로드해 로컬에서 실행한다.

## 속도

`end_to_end_benchmark.py`는 git 태그의 소스와 현재 소스를 임시 디렉터리에 각각 고정하고
동일 Python·의존성으로 별도 프로세스에서 실행합니다. 생성·import·워밍업 시간을 제외한
`analyze`, `mask(tag)`, `mask(stars)` 호출을 각 3회 측정합니다. 출력과 정확한 구간도 검증합니다.
일반 문서 전체에 이 표본의 배율을 적용할 수 없습니다.

**합성 휴대전화 1,000개·21,000자, Python 3.12.13, Presidio 2.2.364, 3회 중앙값:**

| API | v0.2.0 | 개발 버전 | 배율 |
|---|---:|---:|---:|
| analyze | 947.61ms | 106.90ms | 8.86배 |
| mask(tag) | 1,141.83ms | 105.75ms | 10.80배 |
| mask(stars) | 1,162.84ms | 107.03ms | 10.86배 |

```bash
uv run python benchmarks/end_to_end_benchmark.py --output /tmp/end-to-end.json
uv run python benchmarks/performance_benchmark.py --output /tmp/intervals.json
```

전체 API의 원본 표본·환경·소스 해시는 [end-to-end.json](../benchmarks/results/end-to-end.json),
중복 제거와 겹침 처리만 분리한 측정은 [performance.json](../benchmarks/results/performance.json)에
있습니다. 구간 알고리즘의 개선 배율을 전체 탐지 배율로 표시하지 않습니다.

## 남은 범위

커밋 `8e60c89` 시점 검증은 프로젝트 Python 3.12.13에서 실제 로컬 NER·긴 입력·파일명 마스킹을 포함해
**691개 테스트 통과**다. 기본 의존성의 Python 3.10.12·3.11.15·3.13.14는 각각 uv 격리 환경에서
**689개 통과·선택 NER 테스트2개 생략**을 확인했다. Ruff와 `git diff --check`도 통과했다.
`uv build`로 생성한 wheel을 프로젝트 밖 별도 uv 환경(Python 3.13)에 설치해
기본9종·이름주소opt-in·농협4그룹·괄호상세주소·선택마스킹을 확인했다.
이 wheel은 배포와 별개인 로컬 검증 산출물이다.
선택 의존성인 Guardrails/typer의 deprecation 경고는 남아 있습니다.

평가한 데이터가 모든 한국어 업무 문서를 대표하지는 않습니다. 이름·주소와 같은 비정형 PII,
OCR 오류, 다양한 구계좌·가상계좌, 문맥이 멀리 떨어진 식별번호는 별도 데이터와 검증이 필요합니다.
이 문서의 숫자는 재현 가능한 개선 증거이며, 국내 모든 시스템에서 최상의 모델이라는 주장은 아닙니다.
