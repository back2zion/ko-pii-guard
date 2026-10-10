# v7 미탐 6건·오탐 4건 수정 v8

2026-10-10. **회귀로 탈락한 실험이다. 채택하지 않는다.**

## 오류 고정

v7의 새 평가에서 미탐 6개, 잘못된 구간 4개가 남았다. 오류가 발생한 10문장의
원문과 정답을 바꾸지 않고 `test_name_context_v8_regressions.py`의 개별 정확 구간·
별표 마스킹 테스트로 고정했다. v7에서 **10개 모두 실패**했다.
[Red 재현 기록](../benchmarks/results/name-context-v8-tdd-red.json).

오탐 중 하나는 실제 인물과 제목이 함께 있는 문장이었다. 필수 제거 대상에
비인물 문장만 포함하면 이 오류가 선택 조건에서 빠진다. v8에서는 두 문자 판정기의
잘못된 후보를 인물 문장에서도 필수 제거 대상으로 포함한다. 실제 이름은 모든
학습·검증 정답 구간에서 제거 0개를 요구한다. 이 조건은 별도 단위 테스트로 검사한다.

## 두 판정기 학습

고정 E5와 기존 기본 문자 판정 가중치는 보존한다. v7 보조 문자 판정기를 초기값으로
미탐을 학습하고, 기존 비인물 판정기도 잘못된 후보를 함께 학습한다. 추론 구조와
공개 API는 v7과 같으며 이름별 예외나 단어 블랙리스트를 추가하지 않는다.
보조 후보가 이미 받아들인 이름·주소와 겹치면 추가하지 않는다. 긴 문서의 여러
모델 창은 기존 동작을 유지한다.

학습·검증 문맥 표현은 한 번 계산해 두 판정기 학습에 재사용한다.
seed 20261010, 보조 판정기 AdamW 0.0002 / 30 epochs / 확신도 0.99,
비인물 판정기 AdamW 0.0002 / 240 epochs / 제거 임계값 0.99를 미리 정했다.
기존 v6/v7 평가 각 192문장은 개발 회귀 자료로 전환했다.

체크포인트 조건:

1. 보조 문자 판정기: 개발 회귀 384문장의 모든 이름에서 FP/FN 0.
   이 조건을 통과한 모델 중 검증 F1, 검증 FP 수, 최초 시점 순으로 선택한다.
2. 비인물 판정기: 학습·검증의 모든 이름 제거 0, 두 문자 판정기의 학습 오탐
   후보 잔존 0. 검증 음성 후보를 가장 많이 제거한 최초 시점을 선택한다.
3. 최종 공개 API: 개발 회귀 전체 TP468/FP0/FN0과 실제 전체 마스킹을 확인한다.
4. 선택과 개발 공개 API 검사가 끝난 뒤에만 새로운 평가를 실행한다.
   기준선이 찾은 각 정답 구간을 보존하고 새 잘못된 구간은 0이어야 한다.

## 자료 분리

`name_context_v7_regressions.jsonl`은 이전 평가 192문장의 원문·정답을 그대로 보존한다.
`name_context_v8_cases.py`는 기존 모든 v7 분할을 개발 학습 자료로 전환한다.
새 `name_context_v8.jsonl`의 학습 9,192 / 검증 96 / 평가 192문장은 이름·원문·
문맥 ID가 분할 간 겹치지 않는다. 평가에는 이름 234개·비인물 48문장이 있다.
생성 재현성과 기존 주석 보존은 `test_name_context_v8_data.py`로 검사한다.

## 탈락 결과

보조 판정기는 epoch 10, 비인물 판정기는 epoch 75를 선택했다. 개발 공개 API
384문장 TP468/FP0/FN0을 달성했고 기존 미탐 6건·오탐 4건의 10개 테스트도
통과했다. 전체 1,378개 테스트가 통과했지만 새 평가의 기존 정답 구간 1개를 잃었다.
새 192문장의 기준선 TP/FP/FN 229/5/5가 230/5/4로 개선돼도 채택하지 않았다.

잃은 이름은 `v8-evaluation-person-06-011`의 [14,17) 구간이다. 진단 결과 비인물
제거 확률은 0.000238로 낮았고, 추가 학습한 이름 보조 판정기가 해당 구간을
확신도 0.99 이상으로 찾지 못한 것이 원인이었다. v7 이름 보조 판정기는 같은 구간을
0.998169 점수로 찾았다. 후보 판정기 교체에 따른 망각이며 이름 예외로 처리하지 않는다.

[선택과 첫 평가 보고서](../artifacts/name-context-v8/training_report.json),
[기존 검사 Green](../benchmarks/results/name-context-v8-tdd-green.json),
[새 회귀 Red](../benchmarks/results/name-context-v9-tdd-red.json).
v8의 모든 자료는 후속 v9에서 개발로 전환한다.

## 재현

새 출력 경로를 사용해 과거 실험을 보존한다. 첫 명령은 실패 재현용이다.

```bash
KO_PII_TEST_NAME_CONTEXT=1 KO_PII_TEST_NAME_CONTEXT_PATH=artifacts/name-context-v7 uv run --frozen pytest -q tests/test_name_context_v8_regressions.py
OTEL_SDK_DISABLED=true uv run --frozen python scripts/train_name_context_v8.py --output /tmp/name-context-v8-reproduction
KO_PII_TEST_NER=1 KO_PII_TEST_NAME_CONTEXT=1 KO_PII_TEST_NAME_CONTEXT_PATH=artifacts/name-context-v8 OTEL_SDK_DISABLED=true uv run --frozen pytest -q
uv run --frozen python benchmarks/name_context_v8_benchmark.py --split development --output /tmp/name-context-v8-acceptance.json --check
uv run --frozen python benchmarks/name_context_v8_benchmark.py --split evaluation --output /tmp/name-context-v8-nonregression.json --check
```

유한한 합성 자료 검증이다. 실제 업무 문장의 무오류나 실사용 정확도를 보장하지 않는다.
새 평가를 보고 후속 수정을 한다면 그 자료는 개발로 전환하고 다음 평가를 분리한다.
