# 금융 문서형 합성 자료 평가

[Kiii² 프리뷰](https://huggingface.co/datasets/nmixx-fin/kiii-kiii)는 한국어 금융 문서
형식을 본뜬 공개 합성 자료다. 실제 고객 기록은 포함하지 않는다고 제작자가 밝힌다.
라이선스는 CC BY-NC 4.0이며 원문은 이 저장소에 재배포하지 않는다. 이 자료의 점수는
업무 문서 형태에 대한 독립적인 참고 수치일 뿐 실제 사내 문서의 정확도가 아니다.

`scripts/prepare_kiii_evaluation.py`는 전체 1,440문서 파일의 고정 revision
`1aa953c730cd04dac83b496a62b52e700920273b`과 SHA-256
`d57b1cb0b8b67fe48a9b45ee985fb01e63b919147a8527f8231459a68b57456d`을
확인한다. 문서 종류 10개, 변형 수준 `T0`·`T3`, 길이 `1k`·`4k`·`16k`의
각 조합에서 정답을 보지 않고 문서 ID의 SHA-256 순위가 가장 낮은 한 문서를 고른다.
따라서 총 60문서다. 선택이 불가능한 조합이 있으면 실패한다.

기본 구조화 프로필에서는 `rrn`, `foreigner_reg_no`, `business_reg_no`,
`driver_license_no`, `passport_no`, `phone_no`, `email`, `card_no`,
`bank_account_no`만 각각 대응하는 `ko-pii-guard` 9유형의 정답으로 쓴다.
이름·주소를 포함하려면 준비 명령에 `--include-name-address`를 붙이고 평가 명령에도
`KR_NAME KR_ADDRESS`를 포함한다. 같은 문서 안에 있는 지원 범위 밖의 짧은
식별자(고객 ID, 증권 계좌, 법인 번호 등)는 `ignore` 구간으로 표시한다. 이 구간에
완전히 포함되고 채점 정답과 겹치지 않는 예측만 오탐에서 제외한다. 문단 전체를
덮는 상담 내용·거래 정보 주석은 `ignore`에 넣지 않는다. 채점 범위 밖의 개인정보나
자료의 주석 누락이 남을 수 있으므로 precision은 주석 정책의 영향을 받는다.
이름·주소만 따로 검사할 때는 `--only-name-address`를 사용한다. `--length 1k`는
각 문서 종류·변형 수준 조합에서 짧은 문서만 골라 20문서 진단 표본을 만든다.
`--include-hard-negatives`를 붙이면 제작자가 주석한 어려운 음성 구간을
`negative_spans`로 추가한다. 이전 집계와 입력 해시를 그대로 재현하기 위해 기본값은
이 구간을 추가하지 않는다.

```bash
uv run --frozen python scripts/prepare_kiii_evaluation.py \
  --include-hard-negatives --output /tmp/ko-pii-kiii-structured-60-hard-negatives.jsonl
uv run --frozen python benchmarks/private_corpus_benchmark.py \
  --input /tmp/ko-pii-kiii-structured-60-hard-negatives.jsonl \
  --baseline-ref 0e2c7ee --accuracy-only \
  --output /tmp/ko-pii-kiii-structured-60-hard-negative-report.json
```

```bash
uv sync --frozen --extra dev --extra ner
uv run --frozen python scripts/prepare_kiii_evaluation.py \
  --output /tmp/ko-pii-kiii-structured-60.jsonl
uv run --frozen python benchmarks/private_corpus_benchmark.py \
  --input /tmp/ko-pii-kiii-structured-60.jsonl \
  --entities KR_RRN KR_FRN KR_BRN KR_DRIVER_LICENSE KR_PASSPORT \
             PHONE_NUMBER EMAIL_ADDRESS CREDIT_CARD KR_ACCOUNT \
  --baseline-ref 0a85e45136c03772259042df4b2c7a80f3a7e015 \
  --repeats 3 --output /tmp/ko-pii-kiii-structured-60-report.json
```

선택된 문서와 모델 입력은 로컬에만 두고, 저장소에는 원문·문서 ID 없는 집계 보고서만
포함한다. 이 평가셋은 test 분할만 제공하므로 해당 결과를 보고 규칙이나 임계값을
조정하면 이후 동일 자료를 독립 평가로 부를 수 없다.

## 고정 60문서 결과 (2026-10-13)

선택 입력 SHA-256은
`126696072d7409502cae5434fe2cc4784952ecc88d789308de56290c18a78aca`다.
총 644,492자·지원 9유형 정답 1,456구간이다. 기준 `0a85e451`과 현재 소스가
모두 TP 938, FP 111, FN 518로 정확한 유형·구간 F1은 **74.89%**였다.
실제 별표 마스킹으로 전체 가려진 정답은 양쪽 모두 **951/1,456(65.32%)**였다.
정확도 전용 재실행에서도 유형·구간 수치가 같았고, 정답이 있는 문서 60개 중
한 구간 이상 완전히 가리지 못한 문서는 **45개(75%)**였다.
범위 밖 식별자 주석 2,166구간 안의 예측 41건을 오탐에서 제외했다.
모든 선택 문서에 채점 대상 정답이 있어 음성 문서 오탐률은 산출할 수 없다.

| 유형 | 정답 | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|---:|
| KR_RRN | 200 | 147 | 6 | 53 | 83.29% |
| KR_FRN | 29 | 23 | 1 | 6 | 86.79% |
| KR_BRN | 11 | 7 | 0 | 4 | 77.78% |
| KR_DRIVER_LICENSE | 20 | 16 | 1 | 4 | 86.49% |
| KR_PASSPORT | 32 | 15 | 2 | 17 | 61.22% |
| PHONE_NUMBER | 293 | 237 | 29 | 56 | 84.79% |
| EMAIL_ADDRESS | 79 | 55 | 22 | 24 | 70.51% |
| CREDIT_CARD | 120 | 69 | 11 | 51 | 69.00% |
| KR_ACCOUNT | 672 | 369 | 39 | 303 | 68.33% |

계좌·카드·여권의 누락이 두드러진다. 합성값에는 파편화·변형·사전 마스킹 형태가
포함되므로 이 결과를 유효한 완전 식별번호만의 재현율이나 실제 고객 자료의 재현율로
해석하지 않는다. 속도는 세 번의 순차 실행 중앙값으로 기준 `analyze()` 912ms,
현재 845ms였으나 순서·캐시·CPU 상태를 교차 통제하지 않았으므로 속도 개선의
확정 근거로 삼지 않는다. [집계 원본](../benchmarks/results/kiii-structured-60-2026-10-13.json)을
참고한다. [정확도 전용·문서 누락 집계](../benchmarks/results/kiii-structured-60-accuracy-only-2026-10-13.json)는
반복 시간 측정을 하지 않았다.
추가 정규화 최적화는 별도 [직전 커밋 대비 집계](../benchmarks/results/kiii-normalization-incremental-2026-10-13.json)에서
기준 `d36f5bb`와 정확도·문서 누락 수가 같고, 10회 중앙값의 전체 탐지 시간이
855.9ms에서 723.9ms로 줄었다. 비교는 고정 입력·동일 의존성·별도 프로세스에서
수행했지만 순차 실행 순서까지 교차 통제한 속도 실험은 아니다.

2026-10-14의 최신 소스에서 정확도만 다시 검사하면 같은 60문서의
TP/FP/FN은 `0e2c7ee` 기준과 현재가 모두 **938/111/518**이었다.
완전 가림은 **951/1,456**, 누락 문서는 **45/60**으로 같았다.
문자 기준 정답 22,460자 중 가린 문자는 **14,669자**, 정답 밖에서 가린
문자는 **506자**였다. 이 문자는 전체 문서가 아닌 선택된 9유형의 주석
범위에서 계산했다. [정확도·문자 집계](../benchmarks/results/kiii-structured-accuracy-2026-10-14.json)는
반복 시간 측정을 하지 않았다.

같은 60문서를 변형 수준과 길이별로 다시 집계했다. 이는 이미 결과를 확인한
test 자료의 **오류 진단**이며 새 독립 평가가 아니다. 아래 각 묶음은 문서 수가
20~30개이고, 길이·변형 조건의 분모가 달라 직접적인 인과 효과로 해석하지 않는다.

| 묶음 | 문서 | 정답 | TP/FP/FN | 정확 구간 F1 | 완전 가림 |
|---|---:|---:|---:|---:|---:|
| 변형 T0 | 30 | 892 | 787/42/105 | 91.46% | 798/892 |
| 변형 T3 | 30 | 564 | 151/69/413 | 38.52% | 153/564 |
| 길이 1k | 20 | 190 | 125/20/65 | 74.63% | 126/190 |
| 길이 4k | 20 | 470 | 338/29/132 | 80.76% | 339/470 |
| 길이 16k | 20 | 796 | 475/62/321 | 71.27% | 486/796 |

전체 미탐 518구간 중 T3가 **413구간**이다. 이 표본에서는 강한 변형에 대한
강건성이 가장 큰 제한이다. T3 자료를 보고 규칙을 바꾸면 같은 자료의 재측정은
개발 회귀가 되며, 새 독립 자료 없이 실무 개선을 주장할 수 없다.
[집계 원본](../benchmarks/results/kiii-stratified-diagnostic-2026-10-14.json)은
원문·문서 ID 없이 고정 데이터·채점 코드·소스 해시를 기록한다.
로컬 캐시에 원본이 있을 때
`python benchmarks/kiii_stratified_diagnostic.py --check-frozen`으로 재현할 수 있다.

어려운 음성 주석을 추가한 동일 60문서(입력 SHA-256
`cfb8655bbc7ab94d0be9992e53922ea62b28737cdb98d93c9e1fad027ee6e12d`)에서는
기준·현재 버전 모두 **225구간 중 42구간(18.67%)**에 예측이 겹쳤다. 나머지
정확한 유형·구간 TP/FP/FN은 938/111/518로 변하지 않았다.
종류별로는 예시 자리표시자 27/34, 공개 연락처 등 `public_entity` 15/46에
집중됐고, 체크섬 오류 0/49, 주민번호 앞자리처럼 보이는 날짜 0/28,
유사 숫자 0/28, 일반 금액 0/40이었다. 별도 유형별 수치는 고정 원본의
`hard_negatives.type`과 탐지 겹침을 집계한 진단이다.
이 도구의 기본 전화번호 정책은 공개 연락처도 가릴 수 있으므로 15건 전체를
사람의 개인정보 오탐으로 단정하지 않는다. 반대로 예시 자리표시자를 가리는
19건의 계좌 예측과 8건의 전화 예측은 업무상 불필요한 가림이 될 수 있다.
[음성 구간 집계 원본](../benchmarks/results/kiii-hard-negatives-60-2026-10-13.json)은
원문·문서 ID·음성값을 포함하지 않는다.

## 이름·주소 짧은 20문서 진단

같은 선택 규칙에서 길이 `1k`만 고른 20문서의 입력 SHA-256은
`5f5cc0c8575c15d69a82b01a096848e07a99ddcc513d6470b9a711b042863990`이다.
이 문서들은 앞의 60문서에 **포함**되며 독립 표본을 추가한 것이 아니다.
`KR_NAME` 정답 216구간과 `KR_ADDRESS` 정답 17구간만 채점했다.

| 같은 20문서·233구간 | 기준 TP/FP/FN | 현재 TP/FP/FN | 현재 F1 | 현재 완전 가림 |
|---|---:|---:|---:|---:|
| 규칙만 | 27/8/206 | 27/8/206 | 20.15% | 30/233 |
| 규칙 + 고정 E5 NER 0.9 | 154/36/79 | 154/36/79 | 72.81% | 161/233 |

NER 사용 시 이름은 TP 141/FP 32/FN 75(F1 72.49%)였고 주소는
TP 13/FP 4/FN 4(F1 76.47%)였다. 규칙만 사용할 때 이름은
14/6/202, 주소는 13/2/4였다. 즉 NER은 이 표본의 이름 탐지를 늘렸지만
주소 정답 수는 늘리지 못하고 주소 오탐 2건을 더했다. 두 버전 간의 개선은 없었다.
정확도 전용 재실행에서도 점수는 같았고, NER을 켜도 **20문서 중 16문서**에
완전히 가리지 못한 정답 구간이 하나 이상 있었다.
원문 없이 저장한 [규칙 집계](../benchmarks/results/kiii-name-address-rules-20-2026-10-13.json)와
[NER 집계](../benchmarks/results/kiii-name-address-ner-20-2026-10-13.json)를 참고한다.
[문서 누락 집계](../benchmarks/results/kiii-name-address-ner-20-accuracy-only-2026-10-13.json)는
반복 시간 측정을 생략했다.
NER과 규칙 실행의 처리 시간은 별도 부하와 반복 수가 달라 여기서 비교하지 않는다.

## 고정 NER 모델 비교 (2026-10-13)

위 20문서와 별도의 공개 대화형 KDPII 500문장을 같은 현재 소스, 동일 입력,
점수 기준 0.9, CPU 스레드 2개, 반복 시간 측정 없이 다시 평가했다. KDPII에는
이름 정답이 18구간뿐이며, Kiii² 20문서는 위 60문서에 포함된다. 둘 다 실제
업무 자료는 아니다. 비교 보고서의 `corpus_sha256`과
`current_source_sha256`은 각 자료에서 두 모델 간 일치한다.

| 자료·정답 | 모델 | TP/FP/FN | 정확한 유형·구간 F1 | 완전 가림 | 누락 문서 |
|---|---|---:|---:|---:|---:|
| KDPII 500문장·이름 18 | E5 | 15/2/3 | 85.71% | 16/18 | 2/18 |
| KDPII 500문장·이름 18 | KcELECTRA | 0/14/18 | 0% | 8/18 | 10/18 |
| Kiii² 20문서·이름·주소 233 | E5 | 154/36/79 | 72.81% | 161/233 | 16/20 |
| Kiii² 20문서·이름·주소 233 | KcELECTRA | 80/56/153 | 43.36% | 114/233 | 18/20 |

KcELECTRA가 KDPII 이름 8구간을 완전히 가렸는데도 정확한 구간 TP가 0인
것은 경계나 유형을 다르게 예측했기 때문이다. 따라서 정확한 구간 F1만으로
마스킹 효용을 판단하지 않는다. Kiii² 주소 17건에서는 KcELECTRA가
14/3/3(F1 82.35%), E5가 13/4/4(F1 76.47%)였지만, 이름 216건은
KcELECTRA 66/53/150(F1 39.40%)와 E5 141/32/75(F1 72.49%)로
차이가 컸다. 이 표본에서 기본 E5를 교체할 근거는 없다. 두 자료 모두 이미
모델 선택에 사용했으므로 이후 튜닝한 모델의 독립 검증 자료로 재사용하지 않는다.

[KDPII E5](../benchmarks/results/kdpii-names-e5-same-source-2026-10-13.json),
[KDPII KcELECTRA](../benchmarks/results/kdpii-names-kcelectra-2026-10-13.json),
[Kiii² E5](../benchmarks/results/kiii-name-address-e5-same-source-2026-10-13.json),
[Kiii² KcELECTRA](../benchmarks/results/kiii-name-address-kcelectra-2026-10-13.json)에
원문 없는 집계와 고정 모델 revision을 기록했다.

2026-10-14에 채점 코드의 문자 교집합 지표를 추가하고 같은 입력·현재 소스·
모델 기준으로 재실행했다. 각 자료에서 두 모델의 입력 해시, 소스 해시,
채점 코드 해시가 일치한다. 정답 문자는 이름과 주소의 원문 구간 길이를 합한
값이다. 정답 밖 가림은 `ignore`로 표시한 지원 범위 밖 구간의 예측을 제외한다.

| 자료·정답 문자 | 모델 | 정답 문자 가림 | 미가림 문자 | 정답 밖 가림 문자 |
|---|---|---:|---:|---:|
| KDPII 46자 | E5 | 43 | 3 | 5 |
| KDPII 46자 | KcELECTRA | 22 | 24 | 45 |
| Kiii² 1,101자 | E5 | 855 | 246 | 42 |
| Kiii² 1,101자 | KcELECTRA | 738 | 363 | 79 |

정확한 구간 일치뿐 아니라 정답 문자 가림과 정답 밖 가림에서도 이 두 표본은
E5가 낫다. 문자 수는 개인정보의 위험도를 가중하지 않으며, 주석 정책에 따라
정답 밖 문자가 실제로 가리면 안 되는 값인지 달라질 수 있다. 같은 문서 안에
지원 범위 밖 개인정보가 남아 있을 가능성도 있다. 따라서 정답 밖 가림을
실무 오탐 문자 수라고 단정하지 않는다.

[KDPII E5 문자 집계](../benchmarks/results/kdpii-names-e5-characters-2026-10-14.json),
[KDPII KcELECTRA 문자 집계](../benchmarks/results/kdpii-names-kcelectra-characters-2026-10-14.json),
[Kiii² E5 문자 집계](../benchmarks/results/kiii-name-address-e5-characters-2026-10-14.json),
[Kiii² KcELECTRA 문자 집계](../benchmarks/results/kiii-name-address-kcelectra-characters-2026-10-14.json).
