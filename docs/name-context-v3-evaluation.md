# 이름 문맥 판정기 v3: 재현 테스트와 새 평가

## 먼저 고정한 요구사항

v2 평가의 미탐 24개·오탐 구간 5개가 발생한 문장은 총 26개다.
이 문장들의 원래 정답과 ID를 유지한 개별 pytest 회귀 테스트를 먼저 추가했다.
동일한 테스트를 v2에 실행한 Red 결과는 **26 failed / 1 passed**였다.
통과한 하나는 보고서와 테스트 사례 ID가 빠짐없이 일치하는지 확인하는 검사다.
정확한 엔티티·문자 구간과 실제 별표 마스킹을 검사하며 정답을 모델에 맞춰 바꾸지 않는다.

## 개발 자료와 새 평가의 분리

검토한 v2 자료의 모든 분할은 v3에서 개발 학습 자료로 전환했다. 기존 파일과
v1/v2 체크포인트·보고서는 보존한다. 훈련에는 한 글자 이름, 보통명사와 같은 이름,
역할 뒤 조사 경계, 다중 인물, 파일명과 명백한 비인물 문맥을 추가했다.
이름 목록은 데이터 작성용이며 추론 코드에 이름 사전이나 예외 목록을 넣지 않는다.

학습 **6,520문장**, 검증 **84문장**, 새 평가 **280문장**을 추론 전에 고정했다.
모든 정답 이름(두 번째 인물 포함), 문장, 문맥 ID는 분할 간 겹치지 않는다.
새 평가는 별도의 이름과 문맥을 교차 작성한 **합성 평가**이며 실제 현장 정확도를
대표하지 않는다. 평가를 확인한 뒤 이 실행의 모델을 재학습하거나 선택하지 않는다.

고정 설정은 seed 20261010, 50 epoch, CPU 2 threads, 학습률 0.002,
임계값 0.9다. E5 본체와 문자 판정기 구조는 유지한다. 체크포인트는 선언한 기존
개발 회귀 오류 최소화 → 검증 F1 최대화 → 검증 FP 최소화 → 가장 이른 동률 순서로
선택한다. 새 평가 점수는 선택에 쓰지 않는다.

## 재현 명령

```bash
# Red: v2로 고정 회귀 요구사항을 실행하면 실패해야 한다.
KO_PII_TEST_NAME_CONTEXT=1 \
KO_PII_TEST_NAME_CONTEXT_PATH=artifacts/name-context-v2 \
OTEL_SDK_DISABLED=true .venv/bin/python -m pytest -q \
  tests/test_name_context_v3_regressions.py --tb=no

.venv/bin/python benchmarks/name_context_v3_cases.py --check
.venv/bin/python -u scripts/train_name_context.py \
  --data benchmarks/data/name_context_v3.jsonl --epochs 50 \
  --regression-data benchmarks/data/name_context_training.jsonl \
  --regression-data benchmarks/data/name_context_v2_regressions.jsonl \
  --regression-data benchmarks/data/name_address.jsonl \
  --regression-data benchmarks/data/business_korean.jsonl \
  --regression-data benchmarks/data/name_field_boundaries.jsonl \
  --output artifacts/name-context-v3

# Green: 새 v3 체크포인트로 실행한다.
KO_PII_TEST_NER=1 KO_PII_TEST_NAME_CONTEXT=1 \
KO_PII_TEST_NAME_CONTEXT_PATH=artifacts/name-context-v3 OTEL_SDK_DISABLED=true \
  .venv/bin/python -m pytest -q
.venv/bin/python benchmarks/name_context_v3_benchmark.py --split development \
  --output benchmarks/results/name-context-v3-acceptance.json --check
```

학습 출력 경로는 새 디렉터리여야 한다. 기존 실행을 재현할 때에는 `--output`을
다른 빈 디렉터리로 지정한다. 학습 manifest에는 데이터·생성기·라이브러리 소스 해시와
학습 전 고정 시각이 기록된다.

## 실행 결과

고정한 50 epoch 중 **17**을 사전 기준으로 선택했다. 선택 시 검증 이름은
TP/FP/FN = **120/6/0**, 선언한 개발 회귀의 문자 판정기 직접 점수는
**2,141/3/0**이었다. 직접 판정기 점수와 이름 필드 정책까지 적용하는 공개 API의
결과는 구분한다. 학습 시간은 약 527초이며 입력·소스 변경 검사는 통과했다.

Red에서 실패했던 26문장은 v3 공개 API에서 모두 Green이다. 원래 v2 평가
232문장 전체도 TP/FP/FN = **300/0/0**, 전체 마스킹 **300/300**,
음성 문장 오탐 **0/32**다. 실제 v3를 연결한 전체 테스트는 **999 passed**이며
이전 140문장, 파일명·다중 인물·긴 문장 검사도 포함한다. 이는 개발 회귀 결과다.
공개 API 인수 검사도 통과했다. 기존 문맥 808문장, 이름·주소 109문장,
업무 문장 82개, 이름 필드 24문장 모두 FP/FN = 0/0이었다.
[개발 인수 보고서](../benchmarks/results/name-context-v3-acceptance.json)에
각 자료의 정확 구간·실제 마스킹·음성 문장 결과를 기록했다.
숫자 음성 2,200문장도 기본 프로필과 v3 포함 프로필 모두 오탐 0건이었다.
[숫자 회귀 보고서](../benchmarks/results/name-context-v3-numeric-regression.json)는
이름의 자유 문장 정확도와 별도로 해석한다. Ruff와 `git diff --check`도 통과했다.

수정과 선택에 사용하지 않은 새 평가 280문장에는 이름 400개, 음성 문장 40개가 있다.

| 모델 | TP | FP | FN | Precision | Recall | F1 | 완전 마스킹 | 음성 문장 오탐 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 고정 E5 기준선 | 257 | 62 | 143 | 80.56% | 64.25% | 71.49% | 262/400 | 10/40 |
| v2 (동일한 새 자료) | 368 | 31 | 32 | 92.23% | 92.00% | 92.12% | 368/400 | 10/40 |
| v3 | 395 | 31 | 5 | 92.72% | 98.75% | 95.64% | 395/400 | 12/40 |

**새 평가에서 미탐 5개·오탐 구간 31개가 남는다.** 특히 음성 문장 오탐은
12/40으로 E5의 10/40보다 높다. 이름 재현율 향상이 모든 품질 지표의 개선을
뜻하지 않는다. 이전 v2 평가의 FN 24/FP 5와 이 새 평가의 FN 5/FP 31은 자료가
다르므로 직접적인 증감으로 해석하지 않는다. 같은 새 자료에서는 v2→v3의 미탐이
32→5, 오탐 구간은 31→31, F1은 92.12%→95.64%다. 음성 문장 오탐은
10→12로 증가해 여전히 개선이 필요하다. 새 평가를 확인한 뒤 재학습·임계값 조정·
체크포인트 재선택은 하지 않았다.
v3는 명시적 선택 실험이며 기본 모델을 자동 교체하지 않는다.

증거: [Red 보고서](../benchmarks/results/name-context-v3-tdd-red.json),
[Green 보고서](../benchmarks/results/name-context-v3-tdd-green.json),
[최초 새 평가](../benchmarks/results/name-context-v3-holdout.json),
[동일한 새 자료의 v2 비교](../benchmarks/results/name-context-v2-on-v3-evaluation.json),
[학습 manifest](../artifacts/name-context-v3/training_manifest.json),
[전체 학습·최초 평가 기록](../artifacts/name-context-v3/training_report.json).
