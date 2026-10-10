# 이름 문맥 판정기 v2: 알려진 오류 수정과 별도 평가

측정일: 2026-10-10. 대상은 미배포 선택 기능
[`artifacts/name-context-v2`](../artifacts/name-context-v2/README.md)다.

## 수정 범위와 방법

v1의 21개 미탐은 한 글자 이름, 일반명사 동형 이름, 이름 끝 글자와 가운데점 경계에
집중됐다. 파일명 속 이름과 다중 인물 문장에서도 회귀가 있었다. 특정 이름을 탐지하도록
하드코딩하거나 기존 모델의 예측을 무조건 되살리는 대신, 이 유형의 양성·음성 문맥과
정확한 문자 경계를 추가해 같은 문자 판정기를 재학습했다. E5 본체·문자 판정기 구조·
임계값 0.9는 유지했다.

학습 4,488문장, 검증 168문장, 새 평가 232문장을 사용한다. 모든 정답 이름 문자열
(두 번째 인물과 반복 이름 포함), 문장, 문맥 ID는 분할 간 겹치지 않는다. 과거에 확인한
v1의 모든 분할과 기존 개발 자료는 v2에서 학습/선택 자료로 재분류했다. 이전 데이터와
보고서를 덮어쓰지 않고 보존했다. 정답을 모델 예측에 맞춰 변경하지 않았다.

체크포인트 선택은 사전에 고정한 개발 회귀 오류 수 최소화, 검증 exact-span F1 최대화,
검증 FP 최소화, 가장 이른 동률 epoch 순서다. 50 epoch 중 24를 선택했다. 그때 검증은
TP/FP/FN = 202/0/8, 선언한 이름 개발 회귀는 1,841/0/0이었다. 이 값들은 모델 선택에
사용했으므로 독립 평가 성능이 아니다. 새 평가 분할은 선택 후 실행했다.

[학습 manifest](../artifacts/name-context-v2/training_manifest.json)와
[전체 학습 기록·최초 평가](../artifacts/name-context-v2/training_report.json)에
학습 전 동결 시각, 데이터·전체 라이브러리 소스 해시, 고정 설정, epoch별 결과,
체크포인트 해시가 있다. CPU 2 threads를 사용했고 문장을 외부 추론 API로 전송하지 않았다.

## 알려진 v1 오류의 회귀 검사

### 같은 테스트의 Red → Green 재현

v2 개발 후 동일한 회귀 테스트를 수정 전후 체크포인트에 실행하는 사후 검증을 추가했다.
140문장을 개별 pytest 사례로 분리해 실패 ID를 표시하며, 정확한 엔티티·문자 구간과
실제 별표 마스킹을 원래 정답에 대조한다. 테스트용 체크포인트만 환경변수로 바꾸며
제품 기본 모델이나 학습 자료는 변경하지 않는다.

```bash
# Red: 예상된 실패. 기존 v1로 동일한 요구사항을 검사한다.
KO_PII_TEST_NAME_CONTEXT=1 \
KO_PII_TEST_NAME_CONTEXT_PATH=artifacts/name-context-v1 \
OTEL_SDK_DISABLED=true .venv/bin/python -m pytest -q \
  tests/test_name_context_head.py -k closes_all --tb=no

# Green: v2가 기본인 전체 테스트. 실제 NER 통합 테스트도 활성화한다.
KO_PII_TEST_NER=1 KO_PII_TEST_NAME_CONTEXT=1 OTEL_SDK_DISABLED=true \
  .venv/bin/python -m pytest -q
```

Red 결과는 **21 failed / 119 passed**였고, 실패 ID 집합은 보존한 v1 보고서의
21개 미탐 ID와 정확히 일치했다. 기존 E5 대비 회귀 7건은 이 21건의 **부분집합**이며
모두 Red 실패에 포함됐다. 따라서 서로 다른 28건으로 합산하지 않는다.
Green 결과는 실제 NER와 문자 판정기를 활성화한 전체 **971 passed**였으며,
140개 회귀 사례도 모두 통과했다. 테스트 수 증가는 기존 한 테스트의 140개 문장을
개별 사례로 분리한 결과다. Ruff와 `git diff --check`도 통과했다.

공개 `analyze`와 실제 `mask(style="stars")`를 검사했다. 과거 v1 평가 140문장에서
이름 128개의 정확 구간은 107→128, FN은 21→0, FP는 0→0,
전체 마스킹은 107→128이었다. 그중 기존 E5에서 맞았지만 v1에서 틀린 7문장도 모두
정확한 구간과 마스킹을 회복했다. 파일명·다중 인물·긴 토큰 윈도 뒤쪽의 이름도
실제 모델 통합 테스트에 포함한다.

기존 이름·주소·업무·필드 자료에 대해서도 전체 엔티티 프로필의 구간과 마스킹을 검사한다.
[재현 가능한 인수 검사 보고서](../benchmarks/results/name-context-v2-acceptance.json)는
기존 21개 미탐과 7개 회귀의 ID, 남은 ID, 각 개발 자료의 TP/FP/FN, 실제 마스킹,
소스·자료·가중치 해시를 기록한다. `--check`는 알려진 오류 또는 기존 개발 자료의
오탐·미탐이 남으면 실패한다. 이 자료는 **개발 회귀**이며 새 일반화 성능으로 인용하지 않는다.

인수 검사는 통과했다. 과거 문맥 808문장에서도 이름 636/636개의 정확 구간과 전체
마스킹을 확인했고 FP/FN은 0/0이었다. 기존 이름·주소·업무·필드 개발 자료에서도
FP/FN은 0/0이었다. 별도의 숫자 음성 2,200문장은 기본 프로필과 v2 포함 프로필 모두
오탐 0건이었다. [숫자 음성 회귀 보고서](../benchmarks/results/name-context-v2-numeric-regression.json)
에 원문 자료 해시, 모델·체크포인트와 실행 결과를 남겼다.

## 새로 분리한 평가: 같은 232문장·300개 이름

[같은 런타임의 E5/v1/v2 비교](../benchmarks/results/name-context-v2-holdout.json):

| 조건 | 정확 구간 TP / FP / FN | Precision / Recall / F1 | 이름 전체 마스킹 | 음성 오탐 문장 |
|---|---|---|---|---|
| 고정 E5 | 191 / 35 / 109 | 84.51% / 63.67% / 72.62% | 195/300 | 1/32 |
| v1 문자 판정기 | 196 / 3 / 104 | 98.49% / 65.33% / 78.56% | 196/300 | 0/32 |
| v2 문자 판정기 | 276 / 5 / 24 | 98.22% / 92.00% / 95.01% | 277/300 | 1/32 |

v2의 새 평가 집계는 학습 스크립트가 선택 직후 실행한 최초 평가와 일치한다.
이름 20개와 인물 문장 틀 10개를 교차했고 다중 인물·파일명을 포함한다. 음성은 32문장이다.
이름 문자열·문장 틀 분리가 도메인·작성자·의미의 통계적 독립성을 보증하지 않으며,
실사용 표본·전국민 이름 분포·신뢰구간·제품 전반의 정확도로 해석하지 않는다.

**남은 한계:** 새 평가에는 FN 24개와 잘못된 예측 구간 5개가 남았다. 전체 마스킹의
분자가 정확 구간 TP보다 1 큰 것은 과대 구간처럼 이름을 가리면서 exact-span을 틀린
예측이 있기 때문이다. v1보다 새 평가 재현율은 올랐지만 FP도 3→5로 늘었다.
특히 `발표 자료에 적힌 은혜는 사람 이름이 아닌 표제어였다.`는 음성 오탐이다.
오류 ID는 보고서에 그대로 남긴다. 이 결과를 보고 금지어를 추가하거나 정답을 변경하지 않았다.
따라서 v2는 명시적으로 선택하는 실험 기능이며 기본 E5 판정기를 자동으로 교체하지 않는다.
후속 학습에서 이 평가를 사용하면 개발 자료로 취급하고 다시 새로운 평가 자료를 확보한다.

## 재현과 검증

고정 `ner` 의존성과 로컬 E5 캐시를 준비한 뒤:

```bash
uv run --frozen python benchmarks/name_context_v2_cases.py --check
uv run --frozen python benchmarks/name_context_v2_benchmark.py --output /tmp/name-context-v2-acceptance.json --check
uv run --frozen python benchmarks/name_context_v2_holdout.py --output /tmp/name-context-v2-holdout.json
KO_PII_TEST_NER=1 KO_PII_TEST_NAME_CONTEXT=1 OTEL_SDK_DISABLED=true uv run --frozen pytest -q
uv run --frozen ruff check .
```

실제 모델을 포함한 전체 테스트는 **832개 통과**했다. CI에 v2의 알려진 미탐·회귀 인수
검사를 추가했다. [연구 검토](name-context-research.md)의 논문 성능을 가져온 결과가 아니며,
KLUEBERT-CRF·Thunder-DeID·NRB를 재현했다고 주장하지 않는다.
