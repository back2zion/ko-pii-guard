# 비인물 오탐 수정과 이름 회귀 방지 v6

2026-10-10. 선택 기능이며 미배포 상태다. 기본 E5 선택은 바꾸지 않는다.

## 수정

`artifacts/name-context-v6-filter-retry`는 v3의 문자 이름 판정 가중치를 바이트 단위로
보존하고, 고정 E5 표현을 이용하는 별도 학습 판정기로 비인물 후보를 제거한다.
문장·후보 표현과 후보 앞뒤 16문자의 표현을 함께 사용한다. 런타임 이름 목록이나
일반 단어 블랙리스트를 사용하지 않는다. 남긴 후보의 좌표·점수는 그대로 유지한다.
문서가 여러 모델 창으로 나뉘면 이 추가 판정을 적용하지 않아 기존 구간을 보존한다.
따라서 긴 문서의 비인물 오탐 개선은 이 평가로 입증하지 않았다.

v4는 오탐 감소와 함께 기존 정답 8개를 잃어 탈락했다. v5 문자 모델은 검증 이름 1개를
잃어 새 평가 전에 탈락했고, v5 후보 판정기는 새 평가에서 ‘온화’ 이름을 제거해
탈락했다. v6 첫 학습은 개발 오탐 1개가 남아 새 평가 전에 탈락했다. 검사 조건을
완화하지 않고 학습을 다시 수행했다. [회귀 방지 계약](name-context-nonregression.md).

## 기존 비인물 12건과 이전 이름 보존

문제가 됐던 기존 v3 평가 280문장은 개발 회귀 자료로 전환했다. 같은 40개 비인물
문장의 오탐은 **12→0개**이며, 기존에 정확히 찾은 이름 395개는 그대로 보존했다.
잘못된 구간은 31→2개이고 기존 미탐 5개는 남는다. 남은 FP 2개는 인물이 포함된
문장의 추가 잘못된 구간이다. 비인물 해결과 전체 이름 오류 해결을 구분한다.

이후 확인했던 v4 평가 222문장에서도 기존 정답 246개를 보존하며 비인물 오탐은
6/60→0/60이다. 이 자료들은 수정에 사용했으므로 독립 평가 성능으로 제시하지 않는다.

과거 9개 자료 총 2,089문장의 공개 API 인수 검사에서도 기존 정답 이름을 모두
보존하고, 각 자료의 비인물 오탐을 0으로 유지했다.
[개발 인수 보고서](../benchmarks/results/name-context-v6-acceptance.json).

## 학습과 새 평가의 분리

- v5까지 확인한 모든 자료는 개발 학습 자료로 전환했다.
- v6 분할은 학습 8,616 / 검증 96 / 평가 192문장이다. 이름·원문·문맥 ID가
  분할 간 겹치지 않는다. 생성 재현성과 기존 주석 보존을 테스트한다.
- seed 20261010, AdamW 0.001, 240 epochs, 후보 판정 임계값 0.99.
- 학습·검증의 정답 이름 제거 0, 알려진 비인물 후보 잔존 0 조건을 충족하는
  체크포인트 중 검증 음성 제거 수가 가장 큰 최초 시점을 선택했다: epoch 139.
- 새 평가는 선택 후 처음 실행했다. 코드·데이터·기준 가중치 변경 검사도 통과했다.

첫 평가 원본은 [학습 보고서](../artifacts/name-context-v6-filter-retry/training_report.json),
동결 설정은 [manifest](../artifacts/name-context-v6-filter-retry/training_manifest.json)에 있다.

## 수정에 사용하지 않은 새 합성 평가

192문장, 실제 이름 234개, 비인물 48문장. 두 모델 모두 같은 공개 `analyze`와 실제
마스킹을 실행했다.

| 지표 | v3 기준선 | v6 추가 판정 |
|---|---:|---:|
| 정확 이름 구간 TP | 225 | 225 |
| 잘못된 구간 FP | 8 | 0 |
| 이름 미탐 FN | 9 | 9 |
| 완전히 마스킹한 이름 | 225 | 225 |
| 비인물 오탐 문장 | 8/48 | 0/48 |

기존 정답 구간 225개를 각각 비교해 손실 0개, 새 오탐 구간 0개를 확인했다.
집계 수치가 같다는 사실만으로 회귀 통과를 판단하지 않았다. 새 자료의 기존 미탐
9개는 남으며, 이 실험은 그 오류를 고쳤다고 주장하지 않는다. 이 자료를 보고 다시
학습한다면 다음 평가를 별도로 분리해야 한다.

## 통합 검사와 지속 검사

선택 모델을 실제로 연결한 전체 테스트 **1,354개**가 통과했다. 기존 정답 137문장,
역할 이름 회귀, 비인물 40문장과 인물·제목 대조 163문장, 이름·주소·긴 문서·기존
식별번호 및 Guardrails 계약을 함께 검사했다.
[전체 검사 기록](../benchmarks/results/name-context-v6-tdd-green.json)과
[역할 이름 실패 재현](../benchmarks/results/name-context-v6-role-tdd-red.json)을 보존한다.
새 평가를 별도 벤치마크로 다시 실행해도 같은 결과와 구간별 회귀 통과를 확인했다:
[새 평가 검사](../benchmarks/results/name-context-v6-nonregression.json).

## 재현

기존 출력 디렉터리는 덮어쓰지 않는다. 새 출력 경로를 사용한다.

```bash
OTEL_SDK_DISABLED=true uv run --frozen python scripts/train_name_span_filter.py --data benchmarks/data/name_context_v6.jsonl --output /tmp/name-context-v6-reproduction --epochs 240 --local-context
KO_PII_TEST_NER=1 KO_PII_TEST_NAME_CONTEXT=1 KO_PII_TEST_NAME_CONTEXT_PATH=artifacts/name-context-v6-filter-retry OTEL_SDK_DISABLED=true uv run --frozen pytest -q
uv run --frozen python benchmarks/name_context_filter_benchmark.py --output /tmp/name-context-v6-acceptance.json --check
uv run --frozen python benchmarks/name_context_v6_benchmark.py --split evaluation --output /tmp/name-context-v6-nonregression.json --check
```

두 벤치마크는 CI에서 실패 조건으로 실행한다. 기존 이름 손실·새 오탐·비인물 오탐
개선 실패를 성공으로 처리하지 않는다. 유한한 합성 자료 검증이며 실제 업무 문장이나
다른 분포의 무오류를 보장하지 않는다.
