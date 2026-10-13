# 비공개 실무 문서 평가

실제 업무 문서는 저장소 밖이나 Git에서 제외된 `private-eval/` 아래에 둔다. 원문과 정답은
외부 서비스로 전송하지 않는다. 평가 결과에는 원문, 탐지값, 문서 ID, 오류 위치를 기록하지 않는다.
결과 파일도 내부 자료로 취급하고 공개 전 확인한다.

## 준비

실제 입력 경로에서 문서를 유형별로 수집한다. 개인정보가 없는 문서, 형식이 비슷하지만
가리면 안 되는 값, 계좌·카드처럼 헷갈리는 사례를 포함한다. 정답은 두 사람이 독립적으로
검토하고 불일치를 합의하는 편이 좋다. 개발용과 최종 평가용 파일을 문서 단위로 분리하고,
최종 평가용은 규칙이나 임계값을 고르는 데 사용하지 않는다. 중복 템플릿이나 동일 고객
문서가 양쪽에 걸치지 않도록 한다.

UTF-8 JSONL 한 줄이 한 문서다. `start`와 `end`는 Python 문자열의 0-based 문자 위치이며
`end`는 포함하지 않는다. `entities`는 가려야 할 모든 지원 유형의 정답이다. 개인정보가
없는 문서는 빈 목록을 쓴다. 정답 구간은 겹치지 않아야 한다. 아래 예시는 가상 값이다.

한 문서에서 값이 한 번만 나오면 `{"type":"PHONE_NUMBER","surface":"010-1234-5678"}`처럼
`surface`만 적어 위치를 자동 계산할 수 있다. 같은 값이 여러 번 나오면 반드시 위치를
적는다. 위치와 `surface`를 함께 적으면 원문과 일치하는지 검증한다.
지원 범위 밖의 식별정보가 이미 주석된 자료에서는 선택적으로 `ignore` 목록에
`{"start":0,"end":10}` 형태로 적을 수 있다. 정답과 겹치지 않고 제외 구간 안에
**완전히 포함된** 예측만 오탐 분모에서 뺀다. 정답과 겹친 잘못된 구간·유형은 그대로
오탐으로 센다. 넓은 문장·문단 전체를 `ignore`로 지정하면 오탐률을 왜곡하므로
사용하지 않는다. 보고서의 `ignored_spans`와 `ignored_predictions`를 함께 확인한다.
개인정보와 닮았지만 가리면 안 되는 값은 선택적으로 `negative_spans`에
`{"start":0,"end":10}` 형태로 주석한다. 이 구간은 정답·`ignore` 구간과 겹치면
안 된다. 보고서는 해당 구간에 예측이 한 번이라도 겹친 수와 비율을 별도로 기록한다.
실제 가림 정책에서 값 전체가 음성이라는 판정이 확실할 때만 사용한다.

```jsonl
{"id":"sample-001","text":"연락처 010-1234-5678","entities":[{"type":"PHONE_NUMBER","start":4,"end":17}]}
{"id":"sample-002","text":"요청이 접수되었습니다.","entities":[]}
```

기본 평가 대상은 `KR_RRN`, `KR_FRN`, `KR_BRN`, `KR_DRIVER_LICENSE`,
`KR_PASSPORT`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, `CREDIT_CARD`, `KR_ACCOUNT`다.
이름·주소를 규칙만으로 평가하려면 `--entities`에 `KR_NAME`, `KR_ADDRESS`를 추가한다.
선택형 로컬 NER까지 평가하려면 `--entities`에 `KR_NAME` 또는 `KR_ADDRESS`를 포함하고
`--ner`를 추가한다. 모델은 먼저 `python -m ko_pii_guard.ner --download`로 고정 버전을
로컬에 준비해야 한다. 모델 ID와 revision은 보고서에 남기며 두 버전이 같아야 한다.
NER의 기본 점수 기준은 0.9이며 `--ner-threshold`로 동일하게 조정할 수 있다.
로컬 CPU 스레드는 기본 2개이고 `--ner-threads`로 조정한다.
비교 실험에는 `--ner-model e5` 또는 `--ner-model kcelectra`를 사용할 수 있다.
두 버전에서 모델 ID·고정 revision을 같게 확인하고, 서로 다른 모델의 점수는
각각 별도 보고서로 저장한다. 모델 파일은 `python -m ko_pii_guard.ner --download
--model kcelectra`처럼 먼저 로컬에 준비한다.
이름·주소 결과는 기본 구조화 식별번호 결과와 분리해서 해석한다.
운영 설정이 기본 점수 기준 0.4 또는 Unicode 정규화와 다르면 `--threshold` 및
`--no-normalize-unicode` 옵션으로 양쪽 버전에 같은 설정을 적용한다.

```bash
mkdir -p private-eval
chmod 700 private-eval
uv run --frozen python benchmarks/private_corpus_benchmark.py \
  --input private-eval/holdout.jsonl \
  --baseline-ref 0a85e45136c03772259042df4b2c7a80f3a7e015 \
  --output private-eval/report.json
```

개발용 파일도 있다면 `--development-input private-eval/development.jsonl`을 추가한다.
두 파일 사이에 동일한 ID나 Unicode 정규화 후 동일한 문서가 있으면 평가를 중단한다.
값만 다른 동일 템플릿이나 동일 고객의 다른 문서는 자동으로 찾지 못하므로 사람이 분할을
확인해야 한다.
먼저 `--validate-only`를 붙이면 탐지기를 실행하지 않고 형식·구간·중복 분할을
검사한다. 원문 없이 문서 수, 유형별 정답 수, 음성 문서·음성 구간 수, 입력 해시만
출력한다. `--output`을 지정하면 같은 권한 0600 파일에 저장한다. 이 집계에서
예상한 유형의 정답이 0개라면 평가 점수 전에 주석 범위를 확인한다.

기준 버전은 지정한 Git 커밋의 소스를 읽고 현재 버전은 작업 트리의 소스를 읽는다. 두
버전은 같은 Python과 의존성을 쓰는 별도 프로세스에서 실행한다. 시작 시 입력을 권한
0600의 임시 파일로 고정하고 두 프로세스가 같은 복사본을 읽는다. 정상 종료하면 임시
파일을 삭제한다. 보고서의
`corpus_sha256`이 같으면 같은 입력을 평가한 것이다. 속도는 준비·import·생성 시간을
빼고 전체 문서에 대한 `analyze()`와 `mask(style="stars")` 각각 5회 중앙값이다. 정답의 유형·시작·끝이 모두 같아야
TP이며, 실제 `stars` 결과가 정답 구간 전체를 가렸는지는 별도 기록한다. 정답이 없는
문서에서 탐지가 한 건이라도 나온 문서 수도 기록한다. `mask_coverage_by_entity`는
유형별 완전 가림률이며, `wrong_type_same_span`은 계좌를 카드로 분류하는 것처럼
위치는 맞지만 유형이 다른 경우의 집계다.
`positive_documents_with_unmasked_gold`는 정답이 있는 문서에서 한 구간이라도
완전히 가리지 못한 문서 수다. 실제 전달·보관 단위가 문서라면 구간별 가림률과
함께 이 문서 단위 누락률을 우선 확인한다.
`covered_gold_characters`는 정답 문자와 탐지 구간의 교집합이고,
`uncovered_gold_characters`는 가리지 못한 정답 문자 수다.
`overmasked_characters`는 정답 밖에서 가린 문자 수다. 이 문자 지표는 유형을
무시하고 원문 문자 좌표에서 계산하므로 경계가 조금 넓거나 좁은 예측을
진단하는 데 쓴다. 정답 주석 범위 밖의 개인정보와 사전 마스킹된 별표는
개인정보 여부를 확인할 수 없으므로 지표 해석 시 별도로 검토한다.
`negative_span_hit_rate`는 따로 주석한 어려운 음성 구간에서의 탐지 비율이다.
문서 전체 오탐률 및 일반 문서의 precision과 분모가 다르므로 합쳐 해석하지 않는다.
새 보고서에는 채점 스크립트의 `scorer_sha256`과, NER을 사용했다면 PyTorch·
Transformers·Tokenizer·Safetensors 버전도 남긴다. 실행 중 채점 스크립트가
바뀌면 결과를 폐기한다. 이전 보고서에 이 항목이 없어도 이전 점수의 채점 정의는
당시 사용한 커밋과 문서에서 확인한다.
정확도만 필요하면 `--accuracy-only`를 붙인다. 이 경우 정답 비교와 실제 별표 가림은
그대로 수행하고 워밍업·반복 시간 측정은 생략하며 시간 필드는 `null`로 남긴다.

평가 자료의 유형별 정답 개수와 음성 문서 개수를 함께 확인한다. 정답이 드문 유형의
좋은 총점은 그 유형의 성능을 보증하지 않는다. 목표 수치와 허용 가능한 가림 누락을
업무 위험에 맞춰 먼저 정한 뒤 최종 평가 결과로 배포 여부를 판단한다.
