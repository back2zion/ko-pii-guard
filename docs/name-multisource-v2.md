# 출처별 검증과 구간 존재 판정 v2

상태: **채택하지 않음.** `quality_gate_passed: false`가 맞는 결과였다.
`selective:addition=1.0:removal=0.9999` 정책은 KDPII·WikiTree heldout에서
baseline E5와 완전히 동일하고(개선 없음), KLUE NSMC heldout에서 기존 정답
구간 1건("가르시아")을 잘못 제거하는 회귀만 남긴다. 아래는 한 번 틀리게
내렸던 "운영 배포 결정"과 그 철회 기록이다 — 삭제하지 않고 무엇이 왜 틀렸는지
남긴다.

## 운영 배포 결정 — 철회 (원래 승인은 틀린 비교에 근거함)

**원래 결정 (2026-10-10, 철회됨)**: 승인자 back2zion이 이 체크포인트를
"KDPII heldout PS_NAME F1 0.13→0.86, WikiTree/NSMC 거의 유지"라는 순효과
판단으로 운영상 배포 승인했다. "가르시아" 1건 회귀는 외래 인명 미탐으로 보고
watchlist에 올렸다.

**무엇이 틀렸는가**: "0.13→0.86"은 **이미 거부된 다른 후보**
(`name-generalization-lora-v1`, coverage 디코더, threshold 0.01)를 baseline과
비교한 수치였고, 실제로 채택 검토 중이던 `selective` 정책과는 무관했다.
`selective` 정책의 실제 동작(`scripts/name_selective_adapter.py`)은
`addition_threshold=1.0`(baseline이 못 찾은 이름은 추가하지 않음)과
`removal_threshold=0.9999`(아주 확실할 때만 baseline의 기존 구간 제거)로
구성된 **보수적 오탐 제거 필터**일 뿐이다. 실제로 KDPII heldout(1,992문장)의
baseline F1과 candidate F1은 PS_NAME 0.8571 = 0.8571,
PS_NAME+별명 0.7128 = 0.7128로 **완전히 동일**했다 — 이 구간에서 제거가 한
건도 일어나지 않았다는 뜻이고, 어떤 개선도 없었다는 뜻이다. WikiTree도
동일(418/418). 반면 NSMC에서는 이 필터가 정답 구간 1건("가르시아")을 잘못
제거해 회귀를 냈다. 즉 실제 순효과는 **이득 0 + 회귀 1건**이며, 이 상태에서는
baseline을 그대로 두는 쪽이 엄밀히 더 낫다. 이 오류는 README용 실측 평가 표를
작성하며 baseline 열과 candidate 열을 나란히 놓다가 발견했다 — 배포 전,
문서화 단계에서 잡힌 것이다.

**철회 내용**:
- 운영 배포 승인을 철회한다. [`artifacts/name-multisource-v2-verified`](../artifacts/name-multisource-v2-verified)는
  기본 런타임으로 채택하지 않는다.
- 게이트 코드는 원래부터 수정한 적이 없다 — `release-decision.json`의
  `quality_gate_passed: false`가 처음부터 맞았다는 것이 이번에 확인됐다.
- watchlist 항목(`klue-ner-v1_dev_00325-nsmc`, "가르시아")은 유지한다. 이 정책
  자체가 이 구간을 잘못 제거한다는 사실은 여전히 유효한 발견이다.
- 앞으로 같은 `selective` 계열 정책을 다시 검토할 때는, 거부된 다른 후보의
  수치가 아니라 **그 정책이 실제로 비교되는 baseline과** 나란히 놓고 판단한다.

[v1](name-generalization-v1.md)의 새 KLUE 성능은 개선됐지만, KDPII의 비인물
오탐 문장은 1/482에서 101/482로 늘었다. 이름+별명 정책에서도 오탐 구간은
112개였다. 인물 문자 손실 가중치 4와 문자 포함 확률 cutoff .01의 조합을
일반적인 확률 보정으로 해석할 수 없다. 이 실패를 개별 단어 예외로 수정하지 않는다.

## 사용한 자료와 남긴 자료

v1에서 이미 사용한 KLUE 2,000개 ID 전체(추론 전 중복으로 제외했던 1개 포함)와
KDPII 500개는 이제 개발 자료다. v1 보고서·가중치·선택 기록은 변경하지 않는다.
[v2 분리 기록](../benchmarks/results/name-multisource-v2-reserve.json)은 남은 ID 중
`SHA256('ko-pii-multisource-v2:<source>:' + ID)` 순위로 KLUE 1,000개와
KDPII 2,000개를 선택했다. 이름 유무·정답·오류는 선택에 사용하지 않았다.
남는 예비 자료는 각각 2,000개와 2,391개다.

이번 추가 학습 준비에서는 전체 KLUE dev / KDPII test 원문의 정확 해시 및
런타임 `normalize_text` 적용 후 해시와 같은 문장을 **학습·검증 쪽에서 제외**한다.
이 과정은 텍스트 해시만 계산하며 미사용 문장의 정답 주석은 해석하지 않는다.
모델 선택이 끝나기 전에 새 평가를 실행하지 않는다. 기본 E5는 KLUE와 KDPII를
학습에 사용했으며 평가 문장의 포함 여부는 독립적으로 감사하지 못했다. 따라서
이 분리는 이번 추가 학습에 대한 분리다. 정확·정규화 문장 중복 제거만으로
원문 기사나 대화 단위의 독립성을 보장하지 않는다.

## 사전 고정한 개발 비교

문자 포함 확률만으로 붙어 있는 글자를 모으는 대신, 모든 합법 BIOES 경로에서
완전한 이름 구간 `[start, end)`가 나타날 확률을 계산한다. 이 구간 확률도 실제
개인정보 확률로 보정됐다는 보장은 없다. 한 글자 이름도 같은 방식으로 평가한다.
`p(span) > cutoff`인 비중첩 구간 중 `sum(p(span) - cutoff)`가 최대인 조합을
선택한다. 현재 연구 구현은 문장 길이의 제곱에 비례하는 구간 후보를 계산한다.
실시간 보호 방식은 encoder를 세 번 실행하므로 장문·속도 검증도 별도로 필요하다.

먼저 v1 가중치를 고정하고 다음 조합만 이미 사용한 개발 자료에서 비교한다.

- 구간 확률 cutoff: .01, .025, .05, .1, .2, .3, .5, .7, .9, .99.
- 원래 logit과 인물 logit에서 `log(4)`를 뺀 제거 실험. 후자는 학습 가중치의
  영향을 점검하는 대조이며 실제 확률 보정 완료를 뜻하지 않는다.
- 기존 E5 구간을 보호하는 방식과 보호하지 않는 방식. 보호 방식은 기존의
  이름 가림을 유지하는 대신 기존 오탐을 제거할 수 없다는 제약을 명시한다.

출처별 기준선보다 F1·이름 전체 가림이 낮거나 비인물 오탐·불필요 문자 수가
높으면 제외한다. 기존 정확 구간 손실, 새 오탐, 새 이름 노출도 각각 검사한다.
모든 출처에서 같기만 한 후보는 개선 후보로 선택하지 않는다. 통과 후보가 없으면
`no_candidate`로 기록하고 미사용 평가를 열지 않는다. 개발 비교에는 실제 공개
`analyze`와 `mask(style="stars")`에 캐시된 모델 출력을 넣는다. 후보가 생긴 경우에는
실시간 모델 추론과 이 재생 결과의 일치부터 확인해야 한다.

첫 비교는 완료했다. 고정 LoRA 가중치의 40개 정책 중 모든 출처의 조건을 통과한
정책은 **0개**였다. [개발 결과](../benchmarks/results/name-multisource-v2-span-development.json)와
[선택 거부 기록](../benchmarks/results/name-multisource-v2-span-decision.json)을 보존했다.
구간 판정과 보정만으로 문제를 해결했다고 주장하지 않는다. 원래 가림을 보호하면
새 노출은 막을 수 있지만 새 오탐은 남았으며, 보호하지 않으면 기존 이름을 놓쳤다.
이 결과로 미사용 평가를 열지 않았다. 첫 비교는 기존 KLUE CUDA 캐시와 KDPII CPU
캐시를 사용한 개발 실험이며, 동일 device의 최종 성능 비교가 아니다.

## 대화 자료를 포함한 재학습

공식 [KDPII v1](https://zenodo.org/records/10968609)의 train/valid를 준비했다.
KDPII 라이선스는 CC-BY-4.0, KLUE는 CC-BY-SA-4.0이다. 원천 라벨을 보존하고
학습·검증 사이 중복과 충돌을 검사한다. 닉네임·계정 구간은 loss에서 무시하며
몰래 비인물 정답으로 바꾸지 않는다. 평가의 이름/별명 정책은 별도로 유지한다.

| 자료 | 학습 | 검증 |
|---|---:|---:|
| 기존 KLUE 실제 문장 | 10,000 | 2,100 |
| KDPII 대화 | 9,850 | 4,975 |
| 기존 합성 자료 | 2,000 | 96 (선택 제외) |

KDPII 학습은 원본 40,109개에서 라벨과 무관한 ID 해시로 10,000개를 먼저
골랐다. 공식 valid 원본은 5,011개다. 두 split에서 평가 원문과 겹친 125개를
제외하고 일치 중복 61개를 제거해 위 분모를 얻었다. 주석 충돌과 형식 오류는
없었다. KDPII 학습에는 이름 385개가 있는 양성 378문장, 검증에는 이름 179개가
있는 양성 175문장과 이름이 없는 4,800문장이 있다.

원천 SHA256:

- train: `3c65f8c7e48196110d14fa58f591f4a0cb54682a5ab26616292ea54ec4bd46fb`
- valid: `3ffa97a75c584161f1096af6d2bbd252aa5e5ed556bdbe8bcb1467a57bc397c1`

시드 20261013, 3 epochs, 배치 64, AdamW 0.0005로 고정한다. 기존 고정 E5
문자 head에서 시작하며 인코더는 업데이트하지 않는다. 인물/O 가중치를 모두 1로
두고, 기사·리뷰·대화 검증의 exact F1에 같은 가중치를 둬 checkpoint를 선택한다.
이 학습 선택 점수와 출처별 공개 API 채택 검사는 별개다. 동일한 원문·정답·문자
어휘·추출 코드를 검증한 기존 특징만 재사용하고, KDPII의 새 특징은 CPU에서
추출했다. 3 epochs 학습을 완료했고 입력 해시 변경은 없었다. 사전 고정한
출처별 평균 F1 기준으로 epoch 1 / threshold .9를 선택했다.

| 선택 checkpoint의 학습 검증 출처 | TP | FP | FN | Exact F1 |
|---|---:|---:|---:|---:|
| KDPII valid | 156 | 23 | 23 | 87.15% |
| NSMC | 552 | 25 | 25 | 95.67% |
| Wikitree | 901 | 20 | 20 | 97.83% |

세 출처 평균은 93.55%다. 이 수치는 checkpoint 선택에 사용한 학습 검증
결과이며 독립 평가나 공개 API의 성능 개선을 뜻하지 않는다. 원본 결과는
[학습 보고서](../artifacts/name-multisource-v2-verified/report.json)에 보존한다.

재학습 후보의 개발 비교에는 학습 가중치를 이미 1로 변경했으므로 추가 `log(4)`
차감 없이 같은 10개 구간 cutoff와 기존 구간 보호 여부만 비교한다. 같은 출처별
조건을 만족한 후보가 없으면 해당 후보도 채택하지 않는다.

이 20개 정책도 모두 탈락했다. [재학습 개발 결과](../benchmarks/results/name-multisource-v2-trained-development.json)와
[선택 기록](../benchmarks/results/name-multisource-v2-trained-decision.json)을 보존한다.
보호 방식의 cutoff .99에서도 새 오탐은 NSMC 1개, Wikitree 2개, KDPII 이름 정책
1개였다. 전체 이름·별명 정책만으로 이름 정책 실패를 덮지 않는다. 모델과 실제
추론의 재생 일치 및 미사용 평가 단계로 진행하지 않았다.

기존 캐시의 기록 형식이 새 검증기와 달라 두 차례 추출 전 중단이 있었다.
원래 학습 소스의 SHA256과 보관본을 확인하고, `main()` 외 코드의 AST 일치 및
문자 특징 코드의 정확 해시를 검증하도록 호환 경로를 추가했다. 캐시 내용이나
이전 가중치를 바꾸지 않았다. 새 실행의 학습 이력 해시는 초기 모델이 사용한
자료까지 포함한 45,930개 원문 해시의 합집합이다. 최종 평가에서 새 학습 자료만
검사하여 초기 모델의 노출을 누락하지 않는다.

## 선택적으로 가림을 변경하는 추가 개발 실험

앞선 60개 정책은 기존 가림을 전부 보존하거나 보호 없이 다시 판정했다.
전자는 기존 오탐을 줄일 수 없고 후자는 기존 이름을 다시 노출할 수 있다.
같은 고정 가중치로, 기존 구간의 **모든 문자가 O인 사건**에 강한 근거가 있을
때만 가림을 해제하는 방식을 추가 점검한다. 기존 구간과 정확히 같은 이름의
확률이 낮다는 사실만으로 해제하지 않는다. 경계가 다른 진짜 이름일 수 있다.
모든 합법 BIOES 경로에서 전체 O 사건의 확률을 구하며 문자별 O 확률의 곱이나
평균으로 근사하지 않는다. 불확실한 구간은 기존 가림을 유지한다.

추가 비교를 실행하기 전에 다음 20개 조합을 고정한다.

- 새 구간 추가 threshold: .99, .999, .9999, 1.0.
- 기존 구간 해제 threshold: .9, .99, .999, .9999, 1.0.
- 각 사건 확률이 threshold를 **초과**할 때만 변경한다. 1.0은 해당 변경을 끈다.
- logit 보정은 0이다. 추가만·해제만·양방향·기준선 동치 대조를 모두 포함한다.

자료는 이미 소모한 동일한 개발 1,500문장이며, 새 평가를 열거나 모델을 다시
학습하지 않는다. 모든 출처의 실제 가림·개별 회귀·집계 제약을 그대로 적용한다.
추가·해제를 모두 끈 동치 대조는 개선 후보로 채택하지 않는다. 이 사건 확률은
출처가 달라도 실제 정답 확률과 같다는 보정 보장이 없으며, 실제 이름 일부와
군더더기가 섞인 경계 오탐은 해제하지 못할 수 있다.

추가 20개 비교에서 `addition=1.0, removal=.9999` 한 정책만 모든 개발 조건을
통과했다. 새 구간 추가를 끈 정책이므로 **미탐 개선은 없다**. 모델 가중치와
이 정책을 고정하고 실추론 검증을 진행한다.
[개발 결과](../benchmarks/results/name-multisource-v2-selective-development.json)와
[선택 기록](../benchmarks/results/name-multisource-v2-selective-decision.json).

| 개발 출처 / 정답 정책 | 문장 | TP | FP (기준선→후보) | FN | 이름 전체 가림 |
|---|---:|---:|---:|---:|---:|
| NSMC | 494 | 384 | 62→62 | 91 | 416/475 |
| Wikitree | 506 | 381 | 64→62 | 52 | 396/433 |
| KDPII 이름 | 500 | 15 | 2→2 | 3 | 16/18 |
| 같은 KDPII 이름+별명 | 500 | 15 | 2→2 | 17 | 16/32 |

Wikitree의 비인물 오탐 문장은 30→28, 불필요 가림 문자는 113→106이었다.
네 출처·정책의 기존 정확 구간 손실, 새 오탐, 구간 기준 새 노출 및 실제 별표
마스킹의 새 노출은 모두 0이다. 모든 변경을 끈 대조의 기준선 동치와 입력 해시
불변도 확인했다. 이름 전용 `score_threshold=0.0` 검사이며 기본값 `0.4`의 정확도나
일반 문장 오류 해결을 뜻하지 않는다.

실추론에서는 두 개발 출처의 모든 문장에 기존 모델과 후보의 `analyze` 및
`mask(style="stars")`를 각각 실행한다. 좌표와 실제 가림이 캐시 비교와 같아야
하며 신뢰도 수치만의 차이는 별도로 기록한다. 이 검사를 모두 통과한 동일한
가중치·정책·CPU 실행 환경에만 새 평가를 허용한다. 원래 예약은 KLUE 1,000 /
KDPII 2,000이며 아래 텍스트 중복 감사 후 실제 평가 대상은 1,000 / 1,992다.
캐시의 기준선 구간은 신뢰도 1.0으로 재구성했고 실추론은 원래
신뢰도를 유지하므로, 이 수치 차이를 CPU/GPU 편차로 해석하지 않는다.
새 평가에서도 출처별 비악화·개별 회귀 검사를 모두 통과하고 최소
한 품질 지표가 실제로 개선돼야 품질 채택 검사를 통과한다. 결과가 모두 같기만
하면 새 개선 근거가 없는 것으로 차단한다. 어떤 결과도 자동 런타임 교체나
모든 일반 문장에 대한 오류 없음의 보장이 아니다.

두 개발 실추론 검사는 모두 통과했다. 기사·리뷰 1,000문장과 대화 500문장의
기존·후보 구간, 실제 가림, 지표가 캐시 비교와 일치했고, 정확·정규화 학습
중복과 실행 중 입력 변경은 없었다. [KLUE 실추론](../benchmarks/results/name-multisource-v2-live-klue-development.json),
[KDPII 실추론](../benchmarks/results/name-multisource-v2-live-kdpii-development.json).

## 새 평가 직전의 텍스트 중복 감사

원래 예약은 ID를 기준으로 분리했다. 새 정답 변환·모델 추론 전에 원문 해시도
감사해 KDPII에서 이미 소모한 문장과 같은 6개 ID를 발견했다. 원래 표본 내부의
중복 6행 중 4행이 이 그룹과 겹쳐, 소모한 문장을 제외한 뒤 추가로 2행을
제거했다. KLUE와 KDPII 사이의 추가 중복은 없었다.

원래 ID 순서에서 이미 소모한 정확·정규화 텍스트를 제외하고, 남은 텍스트의
중복은 최초 ID만 유지한다. **대체 추출·재표집·가중치·threshold 변경은 없다.**
두 독립 계산에서 KLUE 1,000 / KDPII 1,992, 합계 **2,992문장**을 확인했다.
기존 2,000개 KDPII 표본의 결과로 표시하지 않는다. 원래 reserve와 두 개발
실추론 기록은 보존하며 [추론 전 수정 기록](../benchmarks/results/name-multisource-v2-text-amendment.json)으로
제외 이유와 모수를 고정한다.
표준 JSON의 구문을 읽되 ID와 문장 필드만 중복 계산에 사용했으며 PII 주석을
해석하거나 모델·정책 선택에 사용하지 않았다.

새 KDPII 평가의 첫 실행은 추론 전에 원본 주석 중첩 검사에서 중단됐다.
`Data1.680_5_4`에 같은 `CV_POSITION [37,39)` 주석이 두 번 들어 있었고,
기존 변환기는 동일 주석 반복도 중첩으로 거부했다. 원본 BIO와 이름 정답에는
불일치가 없었다. 이 문장을 제외하거나 대체하지 않는다. 원본 주석을 모두
보존하면서 동일한 `(begin, end, label, form)`만 BIO 검증 시 한 번 적용하는
별도 변환기를 구현했다. 다른 중첩·원문·BIO 불일치는 계속 오류로 처리한다.
동일 이름 주석도 정답 span은 한 번만 집계하고 원본 주석 인스턴스는 모두
보존한다. 수정 전 실패를 재현한 뒤 신규 TDD 15개가 통과했다.
[변환 동치 검증](../benchmarks/results/name-multisource-v2-parser-compatibility.json)에서
기존 개발 500문장의 모든 기존 반환 필드가 같았고, 새 1,992문장은 모두 원래
ID 순서로 검증됐다. 원래 실행 manifest와 변환기는 보존하며 새 실행기만
사용한다. 모델·정책·평가 ID·이름 정답을 바꾸거나 문장을 제외하지 않았다.

## 고정 후보의 새 평가 결과

[KLUE 실제 추론 결과](../benchmarks/results/name-multisource-v2-fresh-klue.json)에서
필수 회귀 검사를 통과하지 못했다. 표본은 개발·추가 학습과 정확·정규화 텍스트가
겹치지 않는 1,000문장이다. E5 자체의 사전학습 노출 미감사 제한은 그대로다.
이름 전용 guard threshold는 0.0이며 기본값 0.4의 결과가 아니다.

| 새 출처 | 문장 / 이름 | TP | FP | FN | 이름 전체 가림 | 실제 새 노출 |
|---|---:|---:|---:|---:|---:|---:|
| NSMC | 486 / 449 | 359→358 | 83→82 | 90→91 | 392→391 | 1 |
| Wikitree | 514 / 418 | 375→375 | 55→55 | 43→43 | 396→396 | 0 |

NSMC의 비인물 오탐 문장은 32→31/182로 줄었지만 기존 정확 이름 하나의 가림을
해제했다. 실제 `mask(style="stars")`도 같은 이름을 새로 노출했다. 정확 F1은
80.5836%→80.5399%, 이름 전체 가림은 392/449→391/449였다. Wikitree의 F1
88.4434%와 모든 품질 지표는 같았다. NSMC의 문장 단위 paired bootstrap 500회
F1 차이 95% 구간은 −0.3931~+0.2617%p이며, 통계적 유의성 유무와 별개로
사전 고정한 개별 노출 금지 조건을 위반했다. 개발에서 선택한 `.9999`가 실제
오류 위험을 보장하는 확률이 아니라는 점도 확인됐다. 결과에 맞춰 threshold를
올리거나 해당 문장 예외를 추가하지 않는다.

독립 검토에서 평가 ID·manifest와 동결 입력 119개 해시가 모두 일치했고, 실행 중
소스 변경이나 추가 학습 자료 중복은 없었다. [최종 채택 검사](../benchmarks/results/name-multisource-v2-release-decision.json)는
거부 결과를 보존한다. 일반 미탐·오탐 문제를 해결했다는 성과로 제시하지 않는다.

KDPII는 동일 주석 반복을 처리한 실행에서 기준선 600/1,992 진행 로그까지 남긴
뒤 중단했다. 4코어 WSL에서 모델 추론 두 개와 통합 테스트를 동시에 실행하던 중
사용자가 VS Code 멈춤을 보고했다. 확인 당시 load average는 7.51, 스왑 사용은
약 1.1GiB였다. 이것만으로 편집기 중단의 직접 원인을 확정할 수는 없지만,
작업 부하를 줄이기 위해 대화 추론과 테스트를 종료했고 남은 KLUE 추론도 완료됐다.
[대화 실행 중단 기록](../benchmarks/results/name-multisource-v2-fresh-kdpii-lossless.json)은
완료 지표를 포함하지 않는다. KLUE의 필수 품질 검사가 이미 실패했으므로 대화
추론을 재시작하지 않았다. **완료한 새 평가는 KLUE 1,000문장이며 2,992문장
전체 평가 완료가 아니다.** 원본 예약·중복 감사·두 중단 실행의 manifest를 보존한다.
이번에 정답을 열었던 자료는 후속 모델 선택 시 개발 자료로 취급한다.

소프트웨어 통합 검사 1,726개는 완료·통과했다. 이후 추가한 파서와 수정한 최종
검사기는 관련 테스트 67개가 별도로 통과했다. 이 검사는 단일 CPU·낮은 우선순위로
8.18초에 완료했다. 추가 전체 재실행은 자원 부하로
중단했으므로 전체 1,754개를 한 번에 통과했다고 표시하지 않는다. 이후 무거운
작업은 순차 실행하고 낮은 우선순위와 CPU 사용 제한을 적용한다.

## 재현 명령

아래 명령은 v1 원천·캐시가 준비된 작업 환경을 기준으로 하며 기존 결과를
덮어쓰지 않도록 새 출력 경로를 사용한다. v2 제외 해시 배열은 분리 기록의
`exclusion_hashes`와 같아야 한다.

```bash
.venv/bin/python scripts/prepare_name_multisource_data.py \
  --exclude-text-hashes /tmp/name-multisource-v2-evaluation-exclusion-hashes.json \
  --output /tmp/name-multisource-v2-reproduction

.venv/bin/python scripts/train_name_multisource.py \
  --data-dir /tmp/name-multisource-v2-reproduction \
  --output /tmp/name-multisource-v2-model-reproduction

.venv/bin/python scripts/name_multisource_selection.py \
  --development benchmarks/results/name-multisource-v2-span-development.json \
  --output /tmp/name-multisource-v2-span-decision.json --check
```

마지막 명령은 첫 40개 정책에 대한 선택 거부를 재현하므로 종료 코드 1이 정상이다.
테스트 실패를 무시하거나 후보를 강제 채택하는 옵션은 없다.

최종 거부 기록은 원본 동결 파일·모델 캐시가 기록된 경로에 있는 환경에서 아래
명령으로 다시 검사한다. `evaluated_selected_counts`는 고정 평가 ID 목록의
길이이며 추론 완료 수가 아니다. 대화 보고서의 `evaluation_complete=false`와
빈 품질 결과 때문에 대화 검사도 거부되며, 완료된 KLUE도 별도로 회귀에 실패한다.

```bash
nice -n 15 taskset -c 3 .venv/bin/python scripts/check_name_multisource_release.py \
  --decision benchmarks/results/name-multisource-v2-selective-decision.json \
  --development benchmarks/results/name-multisource-v2-selective-development.json \
  --checkpoint artifacts/name-multisource-v2-verified \
  --reserve benchmarks/results/name-multisource-v2-reserve.json \
  --amendment benchmarks/results/name-multisource-v2-text-amendment.json \
  --live-development \
    benchmarks/results/name-multisource-v2-live-klue-development.json \
    benchmarks/results/name-multisource-v2-live-kdpii-development.json \
  --heldout \
    benchmarks/results/name-multisource-v2-fresh-klue.json \
    benchmarks/results/name-multisource-v2-fresh-kdpii-lossless.json \
  --kdpii-lossless-parser \
  --output /tmp/name-multisource-v2-release-recheck.json --check
```

이 명령도 거부를 재현하므로 종료 코드 1이다. `taskset -c 3`은 이번 4코어 WSL
환경의 CPU 제한이며 다른 환경에서는 사용 가능한 CPU 번호에 맞춰야 한다.

## 최근 연구 확인

2026-10-10에 원문을 다시 확인했다. 최신 모델의 발표 성능을 이 라이브러리의
한국어 성능으로 옮겨 적지 않는다.

- [GLiNER2-PII, 2026-05](https://arxiv.org/html/2605.09973v1)는 구간 추출과
  42개 PII 라벨을 다룬다. 논문에 명시된 합성 학습 언어에는 한국어가 없으며,
  이름을 일반 명사·조직·제품으로 과탐하는 문제도 보고한다. 따라서 바로 교체할
  근거로 삼지 않고 한국어 실제 문장 비교가 필요한 별도 후보로 분류한다.
- [Meddies-PII v2, 2026-10-08](https://arxiv.org/abs/2609.12544v2)는 다국어
  임상 자료와 여러 외부 벤치마크 평가를 보고한다. 초록은 모델·자료를 논문 채택
  후 공개할 예정이라고 명시한다. 이 작업에서 내려받거나 검증한 모델은 아니다.
- [Thunder-DeID, 2025](https://arxiv.org/abs/2506.15266)는 한국어 판결문
  비식별화를 다룬다. 특정 도메인의 결과만으로 일반 문장 성능을 판단할 수 없으므로
  여기서는 기사·리뷰·대화의 출처별 검증을 유지한다.

구간 존재와 문자 경계를 따로 점검하고 모든 출처에 성능 제약을 거는 결정은
위 논문의 성능을 재현했다는 주장이 아니라, 이 저장소의 실제 실패 결과에 근거한
후속 실험 설계다.
