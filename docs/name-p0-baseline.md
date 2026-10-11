# 이름 인식 M0 기준선 (E5, CPU)

`benchmarks/results/name-p0-baseline-e5.json`에서 자동 생성한다. 아래 표는
`uv run --frozen python scripts/render_name_p0_baseline_table.py`가 예측 사이드카
(`name-p0-baseline-e5.predictions.jsonl`)로 모든 개수를 다시 계산해 저장된 JSON과 일치함을 확인한 결과다.
추론을 다시 실행한 것이 아니다. 모든 자료는 이미 정답을 연 **개발 자료**이며 독립 평가가 아니다.

조건: 기본 E5 `FrameByFrame/korean-pii-e5-base` revision `a308c54b`, 이름 임계값 0.9, 공개 `analyze`와
별도 `mask(style="stars")`, 구간 완전 일치, CPU (torch 2.14.1+cpu). 중복 제외 문서는 추론 전에 고정했고 예비 자료로
보충하지 않았다. 후보는 **999/1992 문서 집합**에서 비교한다. 검토 기록은 `name-p0-m0-review.json`.

| 자료 | 정책 | 선정 | 중복 제외 | 평가 | 실패 | 정답 | TP/FP/FN | P | R | F1 | 전체 가림 | 비인물 오탐 문장 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kdpii_fresh_1992 | PS_NAME | 2000 | 8 | 1992 | 0 | 80 | 66/8/14 | 89.19% | 82.50% | 85.71% | 71 | 3/1916 |
| kdpii_fresh_1992 | PS_NAME+PS_NICKNAME | 2000 | 8 | 1992 | 0 | 114 | 67/7/47 | 90.54% | 58.77% | 71.28% | 72 | 2/1883 |
| kdpii_test_500 | PS_NAME | 500 | 0 | 500 | 0 | 18 | 15/2/3 | 88.24% | 83.33% | 85.71% | 16 | 1/482 |
| kdpii_test_500 | PS_NAME+PS_NICKNAME | 500 | 0 | 500 | 0 | 32 | 15/2/17 | 88.24% | 46.88% | 61.22% | 16 | 1/468 |
| klue_dev_1000 | - | 1000 | 0 | 1000 | 0 | 908 | 765/126/143 | 85.86% | 84.25% | 85.05% | 812 | 43/433 |
| klue_dev_999 | - | 1000 | 1 | 999 | 0 | 876 | 743/138/133 | 84.34% | 84.82% | 84.58% | 796 | 54/442 |
| klue_fresh_1000 | - | 1000 | 0 | 1000 | 0 | 867 | 734/138/133 | 84.17% | 84.66% | 84.42% | 788 | 54/432 |
| synthetic_v11_192 | - | 192 | 0 | 192 | 0 | 234 | 141/12/93 | 92.16% | 60.26% | 72.87% | 141 | 0/48 |
sidecar recomputation matches stored JSON for every row

주의: 현재 이름 경로는 E5의 `private_person`만 `KR_NAME`으로 매핑하고 `personal_handle`(별명)은 무시한다
(`src/ko_pii_guard/ner.py`). KDPII의 별명 포함 정책에서 재현율이 낮은 것은 이 범위 때문이며 모델이 별명을 시도하고
놓친 결과가 아니다. 두 정책을 항상 나란히 보고한다.
