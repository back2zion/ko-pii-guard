# v6 미탐 9건 수정과 회귀 방지 v7

2026-10-10. 선택 기능이며 미배포 상태다. 기본 E5 선택은 바꾸지 않는다.

## 수정 대상과 Red

v6의 별도 합성 평가 192문장에서 미탐 9개가 남았다. 해당 원문과 정답 좌표는
변경하지 않고 `tests/test_name_context_v7_regressions.py`의 개별 테스트로 고정했다.
각 문장은 이름의 정확 구간과 실제 별표 마스킹을 함께 검사한다.
기존 v6로 실행해 **9개 모두 실패**했다.
[실패 재현 기록](../benchmarks/results/name-context-v7-tdd-red.json).

이전에 새 평가로 사용했던 v6 전체 자료는 이제 개발 자료다.
`name_context_v6_regressions.jsonl`은 당시 평가 192문장을 그대로 보존한다.

## 기존 구간 보존

v6 문자 판정기와 비인물 판정 가중치를 보존하고 별도 문자 보조 판정기를 학습한다.
기존 결과를 먼저 확정하고, 보조 판정기가 찾은 높은 확신도의 이름 후보에 같은
비인물 문맥 검사를 적용한 다음 기존 구간과 겹치지 않는 후보만 추가한다.
기존 이름·주소의 객체, 좌표, 점수는 교체하지 않는다. 보조 후보끼리 겹치면
점수가 높은 후보를 선택한다. 여러 모델 창에 걸친 긴 문서는 기존 동작을 유지한다.

이름별 예외·이름 사전·일반 단어 블랙리스트를 추가하지 않는다.
오탐 감소로 이름 손실을 상쇄하지 않는 [회귀 계약](name-context-nonregression.md)을 유지한다.
새 오탐이 추가되어도 실패한다.

## 분리한 새 자료와 선택 기준

`name_context_v7.jsonl`: 학습 8,904 / 검증 96 / 평가 192문장.
모든 v6 분할은 학습으로 전환했으며 기존 정답을 그대로 유지한다.
새 분할의 이름·원문·문맥 ID는 서로 겹치지 않는다. 평가에는 이름 234개,
비인물 48문장이 있다. 재현성과 분리 조건은 `test_name_context_v7_data.py`로 검사한다.

v3 문자 판정기에서 보조 판정기만 초기화해 학습한다. 고정 E5 기반,
seed 20261010, AdamW 0.0002, O 가중치 0.25, 30 epochs, 보조 확신도 0.99.
v6의 192개 개발 회귀에서 보조 판정기의 FP/FN이 모두 0인 체크포인트만 허용한다.
그중 검증 F1, 검증 FP 수, 최초 시점 순으로 선택한다. 선택 후에만 새 평가를 실행한다.

## Green과 새 평가

선택 epoch는 **25**다. 알려진 9개 문장의 정확 구간·실제 별표 마스킹 테스트가
**9개 모두 통과**했다. 고정 기준 판정기와 비인물 판정 가중치는 바이트 단위로
같으며, 추가 판정기 가중치는 별도 파일로 보존한다.

첫 새 합성 평가는 이름 234개·비인물 48문장을 포함하는 총 192문장이다.

| 지표 | v6 기준선 | v7 보조 판정 추가 |
|---|---:|---:|
| 정확 이름 구간 TP | 224 | 228 |
| 이름 미탐 FN | 10 | 6 |
| 잘못된 구간 FP | 4 | 4 |
| 완전히 마스킹한 이름 | 224 | 228 |
| 비인물 오탐 문장 | 3/48 | 3/48 |

기준선이 맞힌 **224개 구간을 각각 보존**했고 새 오탐 구간은 **0개**다.
이번 새 자료에서 기존 오탐은 감소하지 않았다. 수정 대상 9건과 새로운 자료의
잔여 미탐 6개는 서로 다른 사례다. 기존 오탐 4개와 새 평가 미탐 6개를 숨기거나
전체 문장 무오류로 주장하지 않는다. 다음 수정을 위해 이 평가를 사용한다면
개발 자료로 전환하고 또 다른 평가를 분리해야 한다.

[첫 평가·선택 보고서](../artifacts/name-context-v7/training_report.json)와
[학습 전 동결 manifest](../artifacts/name-context-v7/training_manifest.json)를 보존한다.

전체 선택 모델 통합 검사 **1,366개**도 통과했다.
[9건 Green](../benchmarks/results/name-context-v7-nine-green.json),
[전체 통합 검사](../benchmarks/results/name-context-v7-tdd-green.json).

과거 10개 자료 **2,281문장**을 기준선과 구간별로 비교한 인수 검사도 통과했다.
이전에 정확히 찾았던 **2,273개 이름·주소 구간 손실 0개**, 새 오탐 구간 0개다.
수정 대상의 원래 192문장 전체는 **TP234 / FP0 / FN0**이며 비인물 48문장 오탐도 0이다.
기존 비인물 12건의 재발도 없었다. 과거 자료는 개발에 사용했으므로 독립 성능으로
제시하지 않는다. [개발 인수 보고서](../benchmarks/results/name-context-v7-acceptance.json).
새 평가 재실행도 첫 평가와 동일하게 통과했다:
[새 평가 구간별 검사](../benchmarks/results/name-context-v7-nonregression.json).

## 검사 명령

```bash
KO_PII_TEST_NAME_CONTEXT=1 KO_PII_TEST_NAME_CONTEXT_PATH=artifacts/name-context-v6-filter-retry uv run --frozen pytest -q tests/test_name_context_v7_regressions.py
OTEL_SDK_DISABLED=true uv run --frozen python scripts/train_name_context.py --data benchmarks/data/name_context_v7.jsonl --output /tmp/name-context-v7-reproduction --initialize-from artifacts/name-context-v3 --preserve-from artifacts/name-context-v6-filter-retry --learning-rate 0.0002 --outside-weight 0.25 --encoding-batch-size 64 --epochs 30 --score-threshold 0.99 --regression-data benchmarks/data/name_context_v6_regressions.jsonl --require-regression-zero
KO_PII_TEST_NER=1 KO_PII_TEST_NAME_CONTEXT=1 KO_PII_TEST_NAME_CONTEXT_PATH=artifacts/name-context-v7 OTEL_SDK_DISABLED=true uv run --frozen pytest -q
uv run --frozen python benchmarks/name_context_v7_benchmark.py --split development --output /tmp/name-context-v7-acceptance.json --check
uv run --frozen python benchmarks/name_context_v7_benchmark.py --split evaluation --output /tmp/name-context-v7-nonregression.json --check
```

첫 명령은 실패 재현용이다. 새 학습은 기존 산출물을 덮어쓰지 않는 출력 경로를 사용한다.
유한한 합성 자료의 회귀·평가이며 실제 문장 전체의 무오류를 보장하지 않는다.
