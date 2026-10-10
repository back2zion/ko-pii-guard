# 미탐 6건·오탐 4건 해결과 이름 판정기 보존 v9

2026-10-10. 선택 기능이며 미배포 상태다. 기본 E5 선택은 바꾸지 않는다.

## 해결한 오류와 실패 재현

기존 v7 평가의 미탐 6개와 오탐 4개가 발생한 10문장을 정확 구간·실제 별표
마스킹 검사로 고정했다. v7에서 **10개 모두 실패**했고 최종 v9에서 모두 통과했다.
원래 192문장 전체는 **TP234/FP0/FN0**, 비인물 48문장 오탐 0개다.
그보다 앞선 v6 평가 192문장도 TP234/FP0/FN0을 유지했다.

[10건 Red](../benchmarks/results/name-context-v8-tdd-red.json),
[최종 전체 Green](../benchmarks/results/name-context-v9-tdd-green.json).

## 회귀 후보 탈락과 원인 수정

v8은 기존 10건을 해결하고 전체 1,378개 테스트를 통과했지만, 새 평가에서
기존 정답 이름 구간 1개를 잃어 탈락했다. 집계 TP/FN 개선으로 상쇄하지 않았다.
[탈락 기록](name-context-v8-evaluation.md).

진단에서는 비인물 필터의 제거 확률이 0.000238로 낮았고, 추가 학습한 이름
판정기가 이전에 찾던 구간을 확신도 0.99 이상으로 찾지 못했다. 기존 v7 보조
판정기는 같은 구간을 0.998169 점수로 찾았다. 해당 문장도 별도 테스트로 고정해
v8에서 실패를 재현했다. [추가 회귀 Red](../benchmarks/results/name-context-v9-tdd-red.json).

v9는 **기존 v7 기본·보조 이름 판정 가중치를 그대로 보존**하고 v8에서 학습한
이름 판정기를 별도로 추가한다. 기존 보조 판정 결과를 먼저 받아들이므로 새
판정기가 기존 구간을 덮어쓰지 않는다. 두 판정기의 후보에는 v8에서 보강한 같은
비인물 판정기를 적용한다. 인물·제목 혼합 문장의 잘못된 후보도 필수 제거 대상으로
학습했다. 실제 이름은 학습·검증 자료에서 제거 0개를 요구했다.

이름 목록·일반 단어 블랙리스트·이름별 예외를 추론에 넣지 않았다. 기본 E5와
기존 기본 문자 가중치는 변하지 않는다. 여러 창으로 나뉜 긴 문서는 기존 동작을
유지한다. 추가 이름 판정기를 실행하므로 짧은 문장의 추론 연산은 늘어난다.

## 선택과 평가 분리

- v8 보조 판정기는 epoch 10, 비인물 판정기는 epoch 75를 개발·검증 조건으로
  선택했다. [학습 전 설정](../artifacts/name-context-v8/training_manifest.json).
- v8의 첫 평가 결과는 이제 개발 자료로 전환했다. 해당 평가에서 회귀한 기존
  이름을 보존하는 구조를 수정한 후, 이름·원문·문맥 ID가 다른 v9 자료를 동결했다.
- v9는 추가 학습이나 임계값 조정 없이, 고정된 기존 가중치를 조합한다.
  새 검증·평가 자료로 체크포인트나 임계값을 고르지 않았다.
- v9 데이터의 개발 9,480 / 검증 96 / 평가 192문장은 분할 간 이름·원문·문맥 ID가
  겹치지 않는다. 검증 분할은 조합 선택에 사용하지 않았다.
- v6/v7 개발 공개 API 384문장 TP468/FP0/FN0과 v8 개발의 기존 정답 229개 보존·
  새 오탐 0개를 확인한 뒤에만 v9 새 평가를 실행했다.
- v9 새 평가에서는 각 기준선 정답 구간 손실과 새 잘못된 구간이 모두 0이어야 한다.

[조합 전 동결 manifest](../artifacts/name-context-v9/assembly_manifest.json),
[개발 검사와 첫 새 평가](../artifacts/name-context-v9/assembly_report.json).

## 수정에 사용하지 않은 새 합성 평가

192문장, 이름 234개, 비인물 48문장. 기준선은 마지막 채택 모델 v7이다.

| 지표 | v7 기준선 | v9 |
|---|---:|---:|
| 정확 이름 구간 TP | 228 | 229 |
| 잘못된 구간 FP | 3 | 2 |
| 미탐 FN | 6 | 5 |
| 완전히 마스킹한 이름 | 228 | 229 |
| 비인물 오탐 문장 | 2/48 | 2/48 |

기존 정답 **228개를 각각 보존**, 새 오탐 구간 **0개**를 확인했다.
수정 대상이었던 과거 미탐 6건·오탐 4건은 모두 해결했지만 이 새 자료에는
다른 미탐 5개와 오탐 2개가 남는다. 전체 무오류나 실사용 정확도로 일반화하지 않는다.
다음 수정을 위해 이 평가를 사용한다면 개발로 전환하고 다음 평가를 분리해야 한다.

전체 선택 모델 통합 검사 **1,381개**가 통과했다. 새 판정기의 망각·잘못된 짧은
구간이 기존 구간을 덮어쓰지 않는 실제 NER 단위 검사도 포함한다. CI는 기존
정답 손실·새 오탐과 알려진 10건의 재발을 실패로 처리한다.

과거 12개 자료 **2,665문장**을 기준선과 구간별로 비교한 인수 검사도
통과했다. 기존 정답 이름·주소 **2,765개 손실 0개**, 새 오탐 구간 0개다.
v3/v4 개발 자료에 각각 남았던 FP 2개도 모두 제거했다. 다른 과거 미탐은 보고서에
그대로 기록하며 과거 자료의 오류가 전부 해결됐다고 주장하지 않는다.
[개발 인수 보고서](../benchmarks/results/name-context-v9-acceptance.json).
새 평가를 별도로 재실행해도 첫 평가와 동일하게 통과했다:
[새 평가 재검사](../benchmarks/results/name-context-v9-nonregression.json).

## 재현

새 출력 경로를 사용해 기존 산출물을 보존한다. 최초 두 명령은 실패 재현용이다.

```bash
KO_PII_TEST_NAME_CONTEXT=1 KO_PII_TEST_NAME_CONTEXT_PATH=artifacts/name-context-v7 uv run --frozen pytest -q tests/test_name_context_v8_regressions.py
KO_PII_TEST_NAME_CONTEXT=1 KO_PII_TEST_NAME_CONTEXT_PATH=artifacts/name-context-v8 uv run --frozen pytest -q tests/test_name_context_v9_regression.py
OTEL_SDK_DISABLED=true uv run --frozen python scripts/train_name_context_v8.py --output /tmp/name-context-v8-reproduction
OTEL_SDK_DISABLED=true uv run --frozen python scripts/assemble_name_context_v9.py --output /tmp/name-context-v9-reproduction
KO_PII_TEST_NER=1 KO_PII_TEST_NAME_CONTEXT=1 KO_PII_TEST_NAME_CONTEXT_PATH=artifacts/name-context-v9 OTEL_SDK_DISABLED=true uv run --frozen pytest -q
uv run --frozen python benchmarks/name_context_v9_benchmark.py --split development --output /tmp/name-context-v9-acceptance.json --check
uv run --frozen python benchmarks/name_context_v9_benchmark.py --split evaluation --output /tmp/name-context-v9-nonregression.json --check
```

v8 학습 명령은과거 탈락 실험을 재현하므로 마지막 새 평가 검사에서 실패가 예상된다.
조합 명령은 저장소의 고정 부모 가중치를 사용한다. 부모 학습 재현과 최종 조합 재현을
구분한다. 추론은 선택 NER 의존성과 명시적으로 내려받은 고정 E5 모델을 사용하며
기본적으로 로컬·오프라인으로 동작한다.
