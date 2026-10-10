# v11: 미탐 5건·오탐 2건 수정과 구간별 회귀 방지

2026-10-10. 명시적으로 선택하는 실험 모델이다. 기본 E5 모델이나 배포 상태를
변경하지 않는다. 합성 자료의 유한한 결과이며 현장 무오류를 뜻하지 않는다.

## 먼저 고정한 실패

v9의 새 평가에서 남았던 미탐 5개는 4문장에, 오탐 2개는 2문장에 있었다.
원래 평가 192문장을 `name_context_v9_regressions.jsonl`로 보존했다.
`tests/test_name_context_v10_regressions.py`의 6개 테스트는 정확한 탐지 범위와 실제
별표 마스킹을 함께 검사한다. 기준 v9에서 6개 모두 실패한 뒤 최종 v11에서 모두 통과했다.

| 원래 사례 | 수정 대상 |
| --- | --- |
| v9-evaluation-person-02-010 | 인물 이름 감사 미탐 |
| v9-evaluation-person-02-015 | 인물 이름 신중함 미탐 |
| v9-evaluation-person-03-013 | 인물 이름 감사 미탐 |
| v9-evaluation-person-06-012 | 인물 이름 믿음·신중함 미탐 |
| v9-evaluation-negative-02-02 | 보통명사 정진 오탐 |
| v9-evaluation-negative-03-02 | 표제어 정진 오탐 |

이 표는 오류 기록이며 추론용 이름 목록이나 예외 규칙이 아니다.

## 탈락한 후보와 수정

첫 v10 필터는 240회 학습 중 필수 오탐을 모두 제거하지 못해 개발 단계에서 탈락했다.
실제 탐지기가 낸 오탐의 학습 가중치를 높인 v10 후보는 기존 5/2건을 해결했고
1,389개 테스트와 과거 2,857문장의 구간별 검사를 통과했다. 그러나 첫 새 v10 평가에서
기존 정답 이름 **공감**의 `[11,13)` 구간을 잃었다. TP/FP/FN 합계는
217/22/17→218/21/16으로 좋아졌어도 후보를 탈락시켰다.
`artifacts/name-context-v10-rejected/training_report.json`에 첫 결과를 보존했다.

이 구간은 `tests/test_name_context_v11_regression.py`로 고정했다. v10 후보에서
실패를 재현했고 v11에서 탐지와 실제 마스킹을 모두 확인했다. 확인한 v10 자료 전체는
개발로 전환했으며 같은 평가로 체크포인트를 다시 고르지 않았다.

첫 v11 필터도 강화된 이름 보존 기준과 필수 오탐 제거를 함께 만족하지 못해
480회 학습 후 개발 단계에서 탈락했다. 새 v11 평가는 실행하지 않았다.
최종 v11은 필터 중간층을 64→128로 늘리고 어려운 표본을 반복 학습한다.
기존 v9의 주 탐지기·보조 탐지기 가중치를 그대로 보존하고, v10에서 학습한 추가
탐지기도 그대로 유지한다. 추론에는 이름 사전, 단어 블랙리스트, 특정 이름 예외,
접미사 잘라내기를 추가하지 않았다. 필터와 보조 탐지기의 적용 범위는 기존처럼
단일 모델 창이며 긴 입력의 기존 동작은 유지한다.

## 데이터 동결과 선택

- v10: 학습 9,768 / 검증 96 / 평가 192문장. 확인한 전체 10,056문장을 v11 개발로 전환.
- v11: 개발 10,056 / 새 검증 96 / 새 평가 192문장, 전체 10,344문장.
- 분할 간 이름·문장·문맥 ID는 겹치지 않는다. 원래 주석은 그대로 복사하며 생성기와
  데이터 분리 테스트로 재현성을 검사한다.
- 코드·데이터·부모 가중치·재사용한 동결 E5 표현의 해시를 학습 전에 기록한다.
  캐시는 기존 9,768개 학습 문장의 표현이며, 추가 개발 288문장과 새 검증 자료는 별도 인코딩한다.
- 모든 학습·검증 이름의 비인물 점수가 **0.5 미만**이고, 모든 필수 개발 오탐은
  실제 추론 기준 **0.99 이상**으로 제거되는 가중치만 선택한다.
- 조건을 통과한 후보 중 검증 음성 구간 제거 수가 가장 큰 가중치를 선택하고,
  동점이면 먼저 나온 가중치를 유지한다. 선택한 필터는 **65회차**다.
- 어려운 이름은 비인물 점수 0.2 이상, 필수 오탐은 0.999 미만일 때 재학습한다.
  새 평가를 체크포인트 선택에 사용하지 않는다.

원문 소스는 각 실험의 `training_manifest.json`과 대응하는 `training_source.txt` 또는
`rejected_training_source.py`로 확인할 수 있다. 실제 실행한 소스·데이터 변경은 없었다.

## 개발과 과거 회귀 결과

원래 v9 평가 192문장은 **TP234/FP0/FN0**이다. v6·v7의 원래 평가도 같은 결과를
유지한다. 이 3개 분할을 합친 공개 API 검사는 576문장, 이름 702개 모두 정확 탐지·마스킹,
미탐·오탐 0개다.

과거 **3,049문장**에서 기준 v9가 정확히 찾은 이름·주소 **3,222개**를 각각 보존했다.
잃은 정답 구간과 새로 생긴 오탐 구간은 모두 0개다. 기준 결과는 동결된 이전 공개 API
보고서를 재사용하고 오류 행은 v9 API로 재확인했다. 새 후보는 모든 행에서 실제
`analyze`와 `mask`를 실행했다. 재구성한 기준 TP/FP 수가 원래 보고서와 일치함도 확인했다.
CI의 독립 벤치마크는 기준과 후보를 전체 행에서 각각 실행한다.

과거 자료 전체가 무오류라는 뜻은 아니다. 예를 들어 개발로 전환한 v10 평가는
TP221/FP2/FN13이다. v4·v8·v5 과거 평가에도 각각 미탐 1개가 남는다.
결과를 중복 자료끼리 합산해 새로운 정확도처럼 제시하지 않는다.

## 수정에 사용하지 않은 새 평가

첫 v11 평가와 독립 평가 명령의 재실행이 같은 결과였다. 추가 학습이나 선택은 없었다.

| 지표 | 기준 v9 | 선택 v11 |
| --- | ---: | ---: |
| 문장 / 정답 이름 구간 | 192 / 234 | 192 / 234 |
| 정확 탐지 TP | 211 | 211 |
| 오탐 FP | 19 | 17 |
| 미탐 FN | 23 | 23 |
| 이름 전체 마스킹 구간 | 212 | 212 |
| 비인물 문장 오탐 | 0 / 48 | 0 / 48 |

기존 정확 구간 211개 손실 **0**, 새 오탐 구간 **0**으로 채택 조건을 통과했다.
이 별도 자료의 **미탐 23개·오탐 17개는 남는다**. 앞선 5/2건 해결과 모든 실제 문장의
무오류는 다르다. 자료가 다른 v9·v10·v11 평가의 점수를 직접 비교해 일반적인 성능
향상으로 주장하지 않는다.

## 검증 기록과 실행

- 전체 실제 모델·Guardrails 통합 테스트: **1,391개 통과**, 실패·오류·건너뜀 0개.
- [기존 6문장 Red](../benchmarks/results/name-context-v10-tdd-red.json),
  [추가 회귀 Red](../benchmarks/results/name-context-v11-tdd-red.json),
  [전체 Green](../benchmarks/results/name-context-v11-tdd-green.json).
- [과거 회귀 결과](../benchmarks/results/name-context-v11-acceptance.json),
  [새 평가 결과](../benchmarks/results/name-context-v11-nonregression.json),
  [선택 및 첫 새 평가 원문](../artifacts/name-context-v11/training_report.json).

```bash
uv run --frozen python benchmarks/name_context_v11_cases.py --check
KO_PII_TEST_NER=1 KO_PII_TEST_NAME_CONTEXT=1 KO_PII_TEST_NAME_CONTEXT_PATH=artifacts/name-context-v11 OTEL_SDK_DISABLED=true uv run --frozen pytest -q
uv run --frozen python benchmarks/name_context_v11_benchmark.py --split development --output /tmp/name-context-v11-acceptance.json --check
uv run --frozen python benchmarks/name_context_v11_benchmark.py --split evaluation --output /tmp/name-context-v11-nonregression.json --check
```

학습 재현에는 동결 E5 표현 캐시가 필요하다. 아래 첫 명령은 v10의 회귀 거부로
종료 코드 1이 예상되지만 캐시를 남긴다. 이미 기록한 평가를 재생하는 명령이며
이를 새로운 미사용 평가로 취급하지 않는다. 출력 폴더는 새 폴더여야 한다.

```bash
uv run --frozen python scripts/train_name_context_v10.py --output /tmp/v10-feature-rebuild
cp /tmp/v10-feature-rebuild/filter_training_cache.pt /tmp/name-context-v10-filter-training-cache.pt
uv run --frozen python scripts/train_name_context_v11.py --output /tmp/v11-reproduced
```
