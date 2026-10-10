# 실제 문장 학습과 문자 경계 복원 실험 v1

**상태: 학습·독립 평가 완료, 후보 채택 거부. KLUE 개선에도 KDPII 오탐 증가와
개별 회귀가 남았다. 기본 런타임에 적용하지 않는다.**

이 실험은 [이전 공동 구간 모델](name-span-experiment-v1.md)이 합성 문장에서는
개선됐지만 외부 문장에서 기본 E5보다 낮은 성능을 보인 문제를 다룬다.
실제 한국어 문장으로 이름 경계를 학습하고, 기존 E5의 인물 확률을 문자 판정에
직접 전달한다. 학습·선택에 사용하지 않은 문장과 실제 마스킹 API로 결과를 확인한다.

## 기준선과 원인 진단

이전 공개 평가의 기본 E5는 1,000문장, 정답 이름 908개에서 정확 구간 TP 765,
FP 126, FN 143, F1 85.05%였다. 예측 구간의 합집합으로 이름 전체를 덮은 경우는
812개이므로 **완전히 덮이지 않은 이름은 96개**였다. 정확한 시작·끝을 맞추지 못한
143개와 이름이 일부라도 노출된 96개는 다른 지표다. 전자는 경계가 과도하게 넓어도
실패로 계산한다. 두 지표 모두 보고하며, 미탐 143개를 모두 완전한 탐지 누락으로
해석하지 않는다. 근거는 [이전 평가 JSON](../benchmarks/results/name-span-v1-external.json)의
`backbone_e5_reference.metrics`다. 당시 전체 가림 수는 구간 합집합으로 계산한 값이며,
이번 평가에서는 실제 `mask(style="stars")` 반환값도 별도로 검사한다.

이미 확인한 같은 1,000문장을 일괄 추론하여 단계를 분해한 진단에서는 다음을 확인했다.
이는 진단 실행의 수치이며, 기존에 공개한 문장별 추론 수치를 교체하지 않는다.

| 일괄 추론 진단 단계 | TP | FP | FN | 이름 전체를 덮은 수 |
|---|---:|---:|---:|---:|
| E5 argmax 구간, 임계값 적용 전 | 790 | 227 | 118 | 842 |
| E5 구간, 임계값 0.9 적용 | 770 | 128 | 138 | 817 |
| 공개 분석 파이프라인 적용 | 766 | 126 | 142 | 813 |

일괄 추론 결과와 기존 문장별 결과는 한 구간 차이가 있었다. 최종 비교는 같은
평가 실행에서 동일한 공개 API 호출 방식으로 기준선과 후보를 다시 측정한다.
임계값을 없애면 더 많은 이름을 덮지만 오탐도 늘어나는 것을 확인했으므로,
임계값 인하만으로 해결됐다고 판단하지 않는다.

진단한 정답 경계 중 45개는 이름의 오른쪽 끝이 E5 토큰 내부에 있어 토큰의 끝만
사용하는 출력 방식으로 정확히 표현할 수 없었다. 44개는 조사 결합, 1개는 호칭
결합 사례였다. 이것은 토큰 경계를 넘어 문자 경계를 학습할 직접적인 근거다.
공개 파이프라인의 숫자 포함 이름 거부는 정답 PS 4구간과 비정답 2구간을 제거했다.
여기에는 왕호와 단체명처럼 일반 인물 NER 주석과 개인정보 정책이 다른 사례가
포함된다. 이 수치를 줄이기 위한 개별 명칭 예외는 추가하지 않는다.

## 데이터와 분리

원천은 [공식 KLUE 저장소](https://github.com/KLUE-benchmark/KLUE)의
커밋 `3efd98708a40ff49251fddde35453f8fbb11f536`에 고정했다.
학습 준비 프로그램은 `klue-ner-v1.1_train.tsv`만 읽으며 dev 파일을 읽지 않는다.
KLUE 원문과 파생 자료의 라이선스는
[CC-BY-SA-4.0](https://github.com/KLUE-benchmark/KLUE/blob/3efd98708a40ff49251fddde35453f8fbb11f536/License.md)이다.
출처: Park et al., *KLUE: Korean Language Understanding Evaluation*, NeurIPS 2021.

| 원천 파일 | SHA256 |
|---|---|
| KLUE train | `34b9d3d9f9ce9e064abc6ba4c27af43b4b70f2d6cde62c076a28e8aaa17cc544` |
| KLUE dev | `0f4d5e818f7b82d299c3a87856fc40a706f5943207580ddb252397e387050a54` |
| KLUE License.md | `7abe19ec9bb73b36141b999b861d24ad855e808bafe0f81e84cce28556f6c297` |

[준비 프로그램](../scripts/prepare_name_generalization_data.py)은 원문 공백과 문자
좌표를 유지하고, 전체 6유형 BIO 태그와 이름 전용 BIO 태그를 함께 저장한다.
인물 `PS`만 `KR_NAME` 정답으로 옮긴다. 원본 21,008문장에서 동일 문장·주석의
중복 3행을 제거하고, 같은 문장에 주석이 충돌하는 2그룹 4행을 격리했다.
충돌을 임의의 정답으로 덮어쓰지 않으며, 제거된 ID는 준비 manifest에 남긴다.

동일 문장 그룹을 먼저 정리한 뒤 Wikitree/NSMC 출처별로 ID의 고정 해시 순위에서
10%를 검증 자료로 선택한다. 선택은 이름 유무·길이·모델 오류에 의존하지 않는다.
기존 합성 자료는 `name_context_v11.jsonl`의 train만 사용하며 중복 48행을 제거했다.
기존 합성 validation/evaluation 288문장과 이전 외부 평가 1,000문장의 텍스트 해시는
학습 후보에서 제외한다. 이번 준비에서 실제 겹친 후보 문장은 0개였다.

| 용도 | Wikitree | NSMC | 기존 합성 | 합계 |
|---|---:|---:|---:|---:|
| 학습 | 10,288 | 8,613 | 10,008 | 28,909 |
| 체크포인트·임계값 선택 | 1,143 | 957 | 0 | 2,100 |
| 별도 합성 검증 보고 | 0 | 0 | 96 | 96 |

학습/검증 사이 동일 원문 중복은 0개다. 다만 기사·리뷰 원문 단위의 그룹 정보가
없으므로 문장 분리가 같은 원문 출처의 독립성을 보장하지 않는다. Wikitree와 NSMC
모두 학습과 검증에 들어가므로 이 분리는 **새 도메인 검증이 아니다**.
더구나 E5 배포자는 KLUE train으로 이미 학습했다고 명시한다. 여기의 검증 2,100개는
이번 추가 문자 판정기 학습에서 제외된 자료이며, 기본 인코더까지 미관측이라는 뜻이 아니다.

원문 및 학습 캐시는 `/tmp`에 보관한다. 저장소에는 코드, 집계 결과, ID와 해시를
남기며 대규모 원문을 재배포하지 않는다. Apache-2.0 코드 라이선스와 KLUE 데이터
라이선스는 구분한다.

## 모델과 사전 고정한 학습 예산

[모델 코드](../scripts/name_generalization_model.py)는 고정 E5의 토큰 표현을 원문
문자로 펼친다. 문자마다 E5 은닉 표현, 명시적인 학습 문자 어휘 임베딩, 원래 토큰
안에서의 상대 위치, E5 인물 라벨 확률을 사용한다. 다른 PII 유형의 확률은 이름
판정에서 O 확률로 합친다. 여러 글자를 가진 S 토큰은 문자 B/I/E 확률로 옮기며,
토큰이 덮지 않는 공백은 O 사전 확률을 갖는다.

투영 64차원, 문자 임베딩 32차원, 양방향 LSTM 방향별 64차원으로 문자 O/B/I/E/S
잔차 점수를 계산한다. 최종 점수에는 학습 가능한 배율의 E5 문자 로그 확률을 더한다.
배율 초기값은 0.25다. 문자 어휘는 학습 자료에서만 만들고, 미등록 문자는 UNK로
처리한다. 판정 결과는 합법적인 시작·끝·전이를 강제하는 BIOES Viterbi로 디코딩한다.
구간 점수는 선택한 문자 라벨 확률의 평균이다. 이름 사전이나 임의의 조사 제거 규칙을
이번 판정기에 추가하지 않는다.

[연구용 어댑터](../scripts/name_generalization_adapter.py)는 주소의 기존 토큰 출력과
이름의 문자 표현을 한 번의 E5 순전파에서 얻는다. 긴 문장은 겹치는 창의 문자 점수
중 문맥이 가장 넓은 것을 원문 좌표에 모은 뒤 전체 BIOES 경로를 한 번 계산한다.
주소와 겹치는 이름은 기존 주소 보호 정책에 따라 제외한다. 기본 패키지의 자동
로드 대상으로 등록하지 않으며 명시적인 연구 후보로 실행한다.

| 항목 | 고정값 |
|---|---|
| 기본 모델 | `FrameByFrame/korean-pii-e5-base` |
| 기본 모델 revision | `a308c54b4407819624a5661e31e162a269f39818` |
| 인코더 업데이트 | 없음; 표현과 확률을 고정 |
| 시드 | `20261011` |
| 학습 epoch | 6 |
| 배치 | 64 |
| 최적화 | AdamW, 학습률 0.001, gradient norm 제한 1.0 |
| 문자 손실 | 교차 엔트로피; O 가중치 1, B/I/E/S 가중치 2 |
| 비교할 구간 임계값 | 0.5, 0.7, 0.9, 0.95 |
| 선택 순서 | 실제 문장 검증 exact F1 → 적은 FP → 높은 임계값 → 이른 epoch |
| 합성 검증 96문장 | 선택 완료 후 별도 보고; 선택 점수에 합치지 않음 |

기본 [E5 모델 카드](https://huggingface.co/FrameByFrame/korean-pii-e5-base/blob/a308c54b4407819624a5661e31e162a269f39818/README.md)는
MIT 라이선스, XLM-RoBERTa 기반 약 278M 파라미터, KDPII·KLUE train·합성 문서의
혼합 학습을 명시한다. KLUE validation은 제외했다고 보고하지만 그 학습 이력을
독립 감사한 것은 아니다. 이번 후보도 단일 시드 실험이므로 다중 시드 안정성을
주장하지 않는다.

## 최종 평가 자료를 여는 순서

이전 외부 평가 1,000문장은 이미 분석하고 오류 원인에 사용했으므로 이제
**개발 진단 자료**다. 최종 일반화 성능으로 다시 제시하지 않는다.
공식 dev의 나머지 4,000개 ID 중
`SHA256('ko-pii-generalization-v1:' + ID)` 순위의 앞 1,000개를 새 평가로 고정했다.
[분리 manifest](../benchmarks/results/name-generalization-v1-split.json)는 ID만 읽어
준비했고, 이 시점에는 새 문장 본문과 정답을 파싱하지 않았다. 나머지 3,000개는
이번 평가 대상으로도 선택하지 않았다.

체크포인트·임계값·어댑터를 선택한 뒤 `finalized: true`와 관련 파일 해시를 선택
manifest에 기록한다. [평가 프로그램](../scripts/evaluate_name_generalization.py)은
선택과 해시를 검사한 후 지정한 1,000문장만 읽는다. 평가 도중 원천·코드·가중치가
바뀌면 결과를 승인하지 않는다. 새 평가 결과를 확인한 뒤 같은 자료로 임계값이나
체크포인트를 다시 선택하지 않는다.

실제 실행의 추론 전 중복 검사에서 고정 1,000개 중
`klue-ner-v1_dev_04155-wikitree`가 추가 학습/검증 원문 해시와 일치했다.
[원래 실행](../benchmarks/results/name-generalization-final-v1-heldout.json)은
모델 추론 전에 차단됐으며 그대로 보존했다. 이후에는 **모델·가중치·디코더·
임계값을 전혀 바꾸지 않고**, 해당 중복만 제외한 **999개**를 평가했다.
다른 문장을 보충하거나 재추출하지 않는다. 제외 규칙과 기존 1,000개 ID,
제외 ID, 실제 평가 ID를 별도 manifest에 기록한 뒤 추론한다. 이것은 오류 결과를
확인한 뒤 좋은 사례만 고르는 절차가 아니라, 추론 전 발견한 학습 중복을
최종 평가의 분모에서 제거하는 규약 보완이다. 처음 실행한 파이프라인 소스도
기존 동결 해시와 일치하는 사본으로 보존했다.

기준선은 현재 기본 E5, NER 임계값 0.9다. 이름만 평가하며 공개 guard의 별도
점수 제한은 두 모델 모두 0으로 맞춘다. 각 문장에서 `analyze`와
`mask(style="stars")`를 실제로 각각 호출한다. exact TP/FP/FN/F1, 실제 별표의
이름 전체 가림, 불필요하게 가린 문자 수, 비인물 문장 오탐, 출처별 성능을 보고한다.
라이브러리의 기본 guard 임계값은 0.4다. 여기의 이름 전용 평가 설정과 다르며,
선택된 문자 포함 확률이 낮은 구간은 기본 guard 설정에서 다시 제거될 수 있다.
따라서 아래 수치는 `entities=["KR_NAME"], score_threshold=0.0` 조건의 결과이며
모든 기본 설정이나 식별번호까지 포함한 프로필의 성능으로 해석하지 않는다.
기존 정확 구간 손실·새 오탐·이전에 가렸던 이름의 새 노출을 항목별 회귀로 기록한다.
평균 F1 상승만으로 회귀가 없다고 판단하지 않는다.

차이는 동일 문장을 함께 재표집하는 500회 paired bootstrap으로 95% 구간을 계산한다.
이 단위는 문장이며 원문 문서/리뷰 단위가 아니므로 상관성에 따른 불확실성을 작게
추정할 수 있다. KLUE PS에는 공인·가상 인물도 포함된다. 따라서 결과는 이름 경계와
마스킹 범위의 측정이며 사적 개인정보 여부 판단 또는 모든 업무 도메인의 정확도는 아니다.

## 두 번째 출처 KDPII 평가 규약

대화 문장의 비인물 오탐과 출처 변화도 확인하기 위해
[공식 KDPII Zenodo 레코드 10968609](https://zenodo.org/records/10968609)의
원본 v1 `test.json`을 별도 평가에 사용한다. 원천은 DOI
`10.5281/zenodo.10968609`, 6,124,918바이트로 고정했다.
SHA256은 `d2a4141c567528c7a1255d6325c42e0988ff223e71f296d9ac4fac20b01ed7fe`,
공식 레코드의 MD5는 `08eebd98593c1bad23fbf29053b3f9d2`다.
[준비 프로그램](../scripts/fetch_kdpii_evaluation.py)은 레코드 ID·라이선스·크기·
체크섬을 검증하며, 기존 파일과 다른 내용으로 원천이나 선택 목록을 덮어쓰지 않는다.

전체 4,891개 문장 ID에서 `SHA256('ko-pii-kdpii-v1:' + sent_idx)` 순위의
앞 500개를 선택하고, 선택 ID manifest를 저장한 다음 해당 문장의 주석을 검증한다.
이름 유무·문장 길이·모델 오류로 표본을 고르지 않는다. 원문 문자 배열, BIO 태그,
주석의 시작·끝과 표면형이 일치하는지 검사하며, 실패한 문장을 다른 문장으로
교체하지 않는다. 원문 공백과 좌표를 유지하고 관측된 원천 라벨 33유형을
`source_annotations`에 보존한다.

| 평가 정책 | 이름 정답 구간 | 이름 정답이 없는 문장 | 해석 |
|---|---:|---:|---|
| 주 평가: `PS_NAME` | 18 | 482 | `PS_NAME`만 `KR_NAME`으로 매핑 |
| 별도 확장 평가: `PS_NAME` + `PS_NICKNAME` | 32 | 별도 계산 | 별명 14구간을 포함한 정책 |

`PS_ID`는 두 정책에서 제외한다. 주 평가에서는 별명 예측이 오탐으로 계산될 수
있으므로 확장 정책 결과를 함께 보고하며, 결과를 본 뒤 주석을 바꾸지 않는다.
[KDPII 평가 프로그램](../scripts/evaluate_name_kdpii.py)은 모델마다 각 문장의
공개 `analyze`와 실제 `mask(style="stars")`를 각각 한 번 호출한다. 두 주석 정책은
동일한 예측과 마스킹 결과를 재사용하고, 정확 구간·이름 전체 가림·불필요 문자·
비인물 오탐·개별 회귀·paired bootstrap을 정책별로 계산한다.

이 자료로 추가 학습하거나 후보·디코더·임계값을 선택하지 않는다. 최종 KLUE
평가와 같은 확정 후보 및 선택 manifest를 사용한다. 이번 학습·검증 원문의 UTF-8
SHA256 목록도 동결하고, 평가 원문과 하나라도 같으면 해당 ID를 기록한 뒤 추론을
중단한다. 겹친 문장을 제외하거나 다른 문장으로 다시 뽑지 않는다.

기본 E5 배포자는 학습 원천에 KDPII를 포함했다고 명시한다. 사용한 KDPII 버전과
이 test 문장들의 실제 학습 포함 여부는 독립적으로 확인하지 못했다. 따라서
**이번 추가 학습과의 원문 중복 검사를 통과해도 E5까지 미관측인 외부 자료라고
주장하지 않는다.** 주 평가의 이름은 18개뿐이어서 이름 재현율의 불확실성이 크며,
이 표본은 482개 비인물 문장의 오탐을 함께 확인하는 제한된 출처 비교다.

KDPII 데이터의 [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)과
원래 E5 모델의 MIT, 프로젝트 코드의 Apache-2.0 라이선스는 구분한다. 출처는
Yonsei University HamSaeM Kim's Lab, TSCIENTIFIC 및 공식 레코드의 공동 저자다.
원문과 변환 문장은 `/tmp/ko-pii-kdpii-v1`에 보관하며 저장소에는 대규모 원문을
재배포하지 않는다.

두 모델은 같은 device에서 실행해야 하고 Python·PyTorch·Transformers 버전을
보고서에 기록한다. 인코더 적응 후보의 경우 원래 E5의 prior·주소 계산은 FP32,
적응 은닉 표현은 CUDA BF16으로 계산한다. 같은 device가 같은 계산 정밀도를
뜻하지 않으며, CUDA 결과가 CPU 수치와 동일하다고 가정하지 않는다. 또한 이
정확도 평가에는 지연시간 측정이 없으므로 처리 속도 개선의 근거로 사용하지 않는다.

KDPII 실제 추론과 회귀 판정까지 완료했다. 아래 결과에서 확인되듯 이 출처의
오탐 증가는 후보 채택을 거부하는 사유다.

## 재현

준비·검증은 기존 `.venv`에서 실행한다. 프로젝트 `.venv`의 PyTorch는 CPU용이며
패키지의 의존성 정의는 변경하지 않는다. 학습에는 별도 CUDA 연구 환경
`/tmp/ko-pii-generalization-cuda`와 RTX 3090 24GB를 사용한다. 다른 환경에서는 같은
고정 모델·라이브러리를 설치한 Python 실행 파일을 해당 경로 대신 사용한다.
학습 프로그램은 고정 revision의 모델이 이미 로컬 캐시에 있어야 하며, 입력이
모델 길이를 초과하면 조용히 잘라내지 않고 실패한다.

```bash
.venv/bin/python scripts/fetch_klue_name_evaluation.py
.venv/bin/python scripts/prepare_name_generalization_data.py \
  --exclude-text-hashes /tmp/ko-pii-prior-development-text-hashes.json \
  --output /tmp/ko-pii-name-generalization-reproduction

/tmp/ko-pii-generalization-cuda/bin/python scripts/train_name_generalization.py \
  --data-dir /tmp/ko-pii-name-generalization-reproduction \
  --output /tmp/ko-pii-name-generalization-model-reproduction \
  --cache /tmp/ko-pii-name-generalization-features-reproduction.pt \
  --device cuda --seed 20261011 --epochs 6
```

입력의 제외 해시 파일은 이전 평가 JSON의 `selected_ids`에 해당하는 문장만
`read_selected_cases`로 읽고 원문 UTF-8의 SHA256을 계산한 JSON 배열이다. 이미
준비된 입력은 `/tmp/ko-pii-name-generalization-v1-verified/manifest.json`에 원천·
준비 코드·합성 자료·제외 파일과 출력 JSONL의 해시를 기록했다. 다른 경로에서
재현하면 manifest의 경로 문자열은 달라지지만 같은 입력의 출력 JSONL 내용은 같다.

새 평가 실행에는 선택을 완료한 JSON이 필요하다. 예를 들어 선택된 체크포인트를
가리키는 `candidate`와 파일별 `frozen_sha256`, 선택 근거, `finalized: true`를 담은
파일을 `--selection`에 전달한다. 미선택 상태에서는 새 평가를 열지 않는다.

```bash
.venv/bin/python scripts/check_name_generalization_release.py \
  --development benchmarks/results/name-generalization-final-v1-nonoverlap-development.json \
  --heldout benchmarks/results/name-generalization-final-v1-nonoverlap-heldout.json \
  --kdpii benchmarks/results/name-generalization-final-v1-nonoverlap-kdpii.json \
  --output /tmp/name-generalization-release-decision.json --check
```

위 검사는 완료된 보고서와 동결 파일 해시를 검증하고 채택 거부를 뜻하는 종료 코드
1을 반환한다. 실제 후보는 `name_adapted_adapter:build_ner`, 체크포인트는
`artifacts/name-generalization-lora-v1`, decoder는 `coverage`, threshold는 `0.01`이다.
실행 인자와 동결 해시는 [최종 선택 기록](../benchmarks/results/name-generalization-final-v1-nonoverlap-selection.json)에
있다. 최초 중복 차단 보고서를 검증하는
[999문장 평가기](../scripts/evaluate_name_generalization_nonoverlap.py)와
[전체 실행기](../scripts/validate_name_generalization_pipeline.py)는 기존 산출물을
덮어쓰지 않는다. 실행기에는 이 작업 환경의 원천·캐시 경로가 포함되어 있다.

## 개발 결과와 선택 목적 점검

고정 인코더의 6 epoch 학습이 끝났고, 검증 정확 F1로 epoch 6 / threshold 0.5를
선택했다. 검증 TP/FP/FN은 1468/51/30, 별도 합성 검증은 106/1/2다.
이것은 새 외부 결과가 아니다. 실행 소스 사본은
[학습 산출물](../artifacts/name-generalization-v1/README.md)에 보존했다.
실행 후 현재 훈련 코드의 캐시 키에 문자 특징 함수 해시를 추가했다. 이 보완은
학습 당시 소스 사본과 구분하며 이미 학습한 가중치를 바꾸지 않는다.

기존 개발 1,000문장의 실제 API 결과는 정확 F1 85.05% → 90.04%,
오탐 126 → 47로 좋아졌으나, 실제 이름 전체 가림이 812 → 792로 나빠졌다.
기존에 가리던 이름 33개가 노출됐으므로 이 판정 방식을 채택하지 않았다.
[개발 실행 보고서](../benchmarks/results/name-generalization-v1-development.json).

점수 임계값을 0까지 내려도 이름 전체 가림은 792개로 같았다. 손실 33개는
경계 잘림 4개, 최적 BIOES 경로에 이름 후보가 없는 경우 29개였다.
[단계 진단](../benchmarks/results/name-generalization-v1-threshold-diagnostic.json).

최적 경로 하나 대신 모든 합법 BIOES 경로의 문자별 인물 확률을 합산하는
forward-backward 방법도 비교했다. 이는 경험적으로 보정된 보호 확률이 아니다.
개발 자료에서 cutoff 후보 .01/.025/.05/.1/.2/.3/.5를 비교하고, 이름 전체 가림과
F1은 기준 이상, 비인물 오탐 문장과 불필요하게 가린 글자 수는 기준 이하인 후보 중
전체 가림 최대 → 불필요 문자 최소 → F1 최대 순으로 선택했다. .01의
TP/FP/FN은 803/86/105, 전체 가림 817, 비인물 오탐 문장 18(기준 43),
불필요 문자 108(기준 217)이다. 전체 개선에도 기존 정답 손실 15개·새 오탐 37개·
기존 가림 손실 14개가 남아 엄격한 개별 회귀 검사는 통과하지 못했다.
[개발용 정책 비교](../benchmarks/results/name-generalization-v1-coverage-selection.json).

인코더 마지막 4개 층의 query/value에 rank 8 LoRA를 적용하는 추가 실험은
같은 학습 자료, 고정 시드 20261012, 3 epoch로 제한했다. 기존 E5 확률과 주소는
적응 모듈을 끈 원래 모델에서 얻으며, 이름의 문맥 표현만 적응 경로에서 얻는다.
별도 평가 자료의 오류를 추가 학습하거나 이름·조사 예외를 하드코딩하지 않는다.
3회 학습 중 검증 정확 F1 기준으로 epoch 2 / threshold 0.7을 선택했다.
검증 TP/FP/FN은 1468/44/30, 이름 전체 가림 1474/1498, F1 97.54%다.
학습 루프는 221.24초였으며 입력·코드 변경은 없었다. 이 시간에는 초기 자료·
캐시 준비가 포함되지 않는다. [학습 산출물](../artifacts/name-generalization-lora-v1/README.md).

같은 개발 자료에서 LoRA 후보의 문자 포함 확률 cutoff도 사전에 정한 목록으로
비교했다. 고정 인코더와 LoRA의 최선 정책 중 같은 제약·우선순위로 LoRA / .01을
선택했다. 실제 API 재실행의 TP/FP/FN은 810/89/98, 전체 가림 823/908,
비인물 오탐 19, 불필요 문자 118이었다. 기본 E5 대비 정확 구간 손실 11,
새 오탐 구간 41, 이전에 가리던 이름의 새 노출 11이 남았다.
이 개발 결과와 선택을 먼저 동결한 뒤 새 평가를 실행했으며, 새 평가 점수로
모델·임계값을 다시 선택하지 않았다.

## 재현 테스트와 실행 검증

다음 동작을 회귀 테스트로 고정했다.

- 합법 BIOES 경로를 전부 열거한 작은 입력과 forward-backward 문자 확률의 일치
- 이름 전체 가림이 줄거나 비인물 오탐·불필요 마스킹이 늘면 개발 정책 선택에서 제외
- 서로 다른 문장 길이를 패딩한 학습 배치의 원문 토큰 좌표 및 역전파 보존
- LoRA를 끈 원래 출력 복원, 기본 가중치의 gradient 금지, 체크포인트 키·자료형 검증
- 이름 특징과 주소 판정의 두 경로 분리, 긴 문장 좌표, 앞뒤 공백 문맥, 동시 요청 격리
- 평가 ID 변경·학습 원문 중복·실행 device 불일치·미확정 후보의 새 평가 실행 차단
- `analyze`가 같더라도 실제 별표 마스킹이 다르면 별도 손실로 보고

패딩 배치와 단일 창 앞뒤 공백 오류는 실패 테스트로 먼저 재현한 뒤 고쳤다.
GPU 한 배치 시험에서도 손실 역전파와 기본 인코더 동결을 확인했다.
기본 모델·기존 v11 실제 통합 검사까지 켠 전체 테스트는 **1,534개 통과**했다
(96.80초, 의존성의 deprecation warning 11개).
이는 코드와 기존 회귀 계약의 결과이며, 새 모델이 독립 평가를 통과했다는 뜻은 아니다.

```bash
KO_PII_TEST_NER=1 KO_PII_TEST_NAME_CONTEXT=1 \
KO_PII_TEST_NAME_CONTEXT_PATH=artifacts/name-context-v11 \
  .venv/bin/python -m pytest -q
```

## 최종 결과와 채택 판정

2026-10-10 평가 완료. 아래는 같은 실행에서 측정한 기본 E5 → 고정 LoRA 후보다.
FN은 정확 구간 기준이며, 전체 가림 실패는 실제 별표 결과에서 이름 일부가 남은
경우다. 모든 평가에서 실행 중 코드·원천·가중치 변경은 없었고, 남긴 평가 문장과
이번 학습·검증 원문의 정확 중복은 0개였다.

| 자료 / 정책 | 문장 / 이름 | 정확 F1 | FP | FN | 전체 가림 실패 | 비인물 오탐 문장 | 불필요 마스킹 문자 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 기존 개발 KLUE | 1,000 / 908 | 85.05% → 89.65% | 126 → 89 | 143 → 98 | 96 → 85 | 43 → 19 | 217 → 118 |
| 새 KLUE | 999 / 876 | 84.58% → 89.14% | 138 → 99 | 133 → 92 | 80 → 73 | 54 → 34 | 301 → 171 |
| KDPII / PS_NAME | 500 / 18 | 85.71% → 22.52% | 2 → 116 | 3 → 1 | 2 → 0 | 1 → 101 | 5 → 168 |
| 같은 KDPII / 이름+별명 | 500 / 32 | 61.22% → 25.45% | 2 → 112 | 17 → 11 | 16 → 9 | 1 → 90 | 5 → 151 |

근거: [개발](../benchmarks/results/name-generalization-final-v1-nonoverlap-development.json),
[새 KLUE](../benchmarks/results/name-generalization-final-v1-nonoverlap-heldout.json),
[KDPII](../benchmarks/results/name-generalization-final-v1-nonoverlap-kdpii.json).
KDPII 두 행은 동일 500문장에 서로 다른 정답 정책을 적용한 것이므로 합산하지 않는다.

새 KLUE에서 F1 차이의 문장 bootstrap 95% 구간은 +2.79~+6.57%p였다.
전체 이름 가림률 차이는 -0.77~+2.19%p로 0을 포함한다. Wikitree의 F1은
85.01% → 90.53%, NSMC는 84.16% → 87.83%였다. 전체 수치가 개선됐어도
기존 정확 구간 손실 10개, 새 오탐 구간 45개, **이전에 가리던 이름의 새 노출
12개**가 생겼다. 새 오탐은 전체 FP의 차이와 다른 문장별 회귀 지표다.

KDPII 주 평가의 F1 차이 95% 구간은 -78.58~-47.46%p였다. 전체 가림은
18/18이지만 새 오탐 구간 114개가 생겼다. 이름+별명 정책에서도 새 오탐 110개와
F1 하락이 남으므로 별명 주석 차이만으로 이 실패를 설명할 수 없다.
이름이 드문 대화 출처에 대한 판정 보정 실패를 확인했으며, KLUE에서 선택한 낮은
문자 확률 cutoff의 개선을 다른 출처로 일반화할 수 없었다.

기존 개발 자료에서 고정한 100문장으로 CPU/GPU도 비교했다. 탐지 구간과 실제
별표 결과 차이는 0문장, 점수 차이는 11문장이었다.
[비교 보고서](../benchmarks/results/name-generalization-final-v1-nonoverlap-device-parity.json).
이는 PyTorch 2.9.0에서 CPU FP32 / CUDA 적응 경로 BF16을 비교한 제한된 결과다.
모든 문장·임계값 또는 프로젝트의 다른 PyTorch 버전에서 같다는 보장은 아니다.

[채택 검사](../benchmarks/results/name-generalization-final-v1-nonoverlap-release-decision.json)는
출처별 성능 제약과 개별 회귀 검사 모두를 요구하며 **실패**했다. 따라서 후보를
기본 런타임에 반영하지 않는다. 전체 코드 테스트 통과와 모델 품질 검사 실패를
구분한다. 이 검사는 외부 평가 판정만 담당하며, 후보 자체가 과거의 모든 모델
회귀 계약을 통과했다는 의미도 아니다. 다음 실험에서는 이번 평가를 다시 독립
검증으로 재사용하지 않고, 별도의 미사용 자료로 최종 판정해야 한다.
