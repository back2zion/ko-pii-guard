# 이름 문맥 판정기 v4: 비인물 오탐 회귀 수정

**채택 거부:** 새 평가에서 비인물 오탐은 6/60→1/60으로 줄었지만 이름 미탐이
24→32로 증가했다. 기존 v3에서 맞았던 8문장의 회귀를 별도 테스트로 재현했다.
전체 개발 테스트 통과만으로는 채택할 수 없다. v4를 일반적인 대체 모델로 권장하지 않는다.

## 고정한 문제와 Red

v3의 새 평가에서 비인물 문장 오탐이 v2의 10/40에서 12/40으로 증가했다.
전체 비인물 40문장을 원래 ID·텍스트·정답 그대로 개별 테스트로 고정했다.
공개 `analyze`가 빈 결과를 내고 기본·별표 `mask`가 문장을 보존해야 한다.
v3를 적용한 Red 결과는 **12 failed / 29 passed**이며 실패 ID는 원래 보고서의
12개 비인물 오류 ID와 정확히 일치했다. 통과한 29개에는 사례 완전성 검사 1개가 포함된다.

## 수정과 새 자료 분리

표어·보통명사 해설·용어 풀이 파일 등 비인물 문맥과 동일 단어의 실제 인명 문맥을
함께 보강했다. 제목과 저자가 한 문장에 나오는 혼합 문맥도 추가하고 163개 개발
대조 테스트로 실제 저자만 마스킹하도록 검사한다. 추론 코드의 이름 사전·단어 예외
목록·문장 전체 차단 규칙·일괄 파일명 제외는 추가하지 않는다.

기존 v3의 모든 분할은 학습 개발 자료로 전환한다. 원본 파일·v3 모델·평가 보고서는
보존한다. 새 v4 자료는 학습 **8,038문장**, 검증 **68문장**, 평가 **222문장**이다.
새 평가에는 이름 **270개**, 비인물 **60문장**이 있다. 모든 정답 이름(두 번째 인물
포함), 문장, 문맥 ID는 분할 간 겹치지 않는다. 이 자료는 합성 교차 문맥 평가이며
현장 표본이나 실사용 정확도의 추정치가 아니다.

v3 체크포인트에서 초기화해 학습률 **0.0002**, 비인물 문자 클래스 가중치 **1.0**,
20 epoch, seed 20261010, CPU 2 threads로 미세조정한다. E5는 고정하며 문자
판정기의 구조와 추론 임계값 0.9는 유지한다. 학습 manifest는 초기 모델·자료·
생성기·라이브러리 소스 해시와 설정을 학습 전에 기록한다.
선택은 선언한 개발 회귀 오류 최소화 → 검증 F1 최대화 → 검증 FP 최소화 →
가장 이른 동률 순서다. 새 평가 결과는 선택에 사용하지 않는다.

## 재현

```bash
# Red: 알려진 12개 실패를 재현한다.
KO_PII_TEST_NAME_CONTEXT=1 KO_PII_TEST_NAME_CONTEXT_PATH=artifacts/name-context-v3 \
OTEL_SDK_DISABLED=true .venv/bin/python -m pytest -q \
  tests/test_name_context_v4_negatives.py --tb=no

.venv/bin/python benchmarks/name_context_v4_cases.py --check
OTEL_SDK_DISABLED=true .venv/bin/python -u scripts/train_name_context.py \
  --data benchmarks/data/name_context_v4.jsonl --epochs 20 \
  --initialize-from artifacts/name-context-v3 --learning-rate 0.0002 --outside-weight 1 \
  --regression-data benchmarks/data/name_context_training.jsonl \
  --regression-data benchmarks/data/name_context_v2_regressions.jsonl \
  --regression-data benchmarks/data/name_context_v3_regressions.jsonl \
  --regression-data benchmarks/data/name_address.jsonl \
  --regression-data benchmarks/data/business_korean.jsonl \
  --regression-data benchmarks/data/name_field_boundaries.jsonl \
  --output artifacts/name-context-v4

# Green: 기존 이름·파일명·긴 문장 계약도 같은 v4로 검사한다.
KO_PII_TEST_NER=1 KO_PII_TEST_NAME_CONTEXT=1 \
KO_PII_TEST_NAME_CONTEXT_PATH=artifacts/name-context-v4 OTEL_SDK_DISABLED=true \
  .venv/bin/python -m pytest -q
.venv/bin/python benchmarks/name_context_v4_benchmark.py --split development \
  --output benchmarks/results/name-context-v4-acceptance.json --check
```

재실행 시 `--output`에 별도의 빈 디렉터리를 지정한다. 원본을 덮어쓰지 않는다.
