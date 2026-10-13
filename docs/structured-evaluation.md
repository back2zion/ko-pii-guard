# 구조화 JSON 레코드의 비공개 정확도 평가

`benchmarks/structured_corpus_benchmark.py`는 `mask_json()`의 **필드 전체**
가림을 이전 커밋과 현재 소스에서 비교한다. 속도는 측정하지 않는다. 일반
문서의 부분 구간 평가와 형식이 다르므로
[문자열 문서 평가](private-evaluation.md)와 점수를 합치지 않는다.

한 줄은 하나의 UTF-8 JSON 객체다. `record`는 마스킹할 JSON 객체 또는 배열,
`targets`는 필드 전체가 한 유형의 개인정보인 값의 주석이다. `path`는 최상위에서
값까지의 객체 키·배열 인덱스 목록이다. 문자열 또는 정수 값만 정답으로 지정할 수
있다. `surface`는 선택 항목으로, 쓰면 원본 값과 완전히 일치해야 한다. 같은
문서 안의 개인정보가 있는 필드는 빠짐없이 주석해야 한다.
앞자리 0이나 구분자를 보존해야 하는 식별번호는 JSON 숫자가 아니라 문자열로
저장한다. 부동소수점 식별번호는 자릿수가 손상될 수 있어 평가 입력으로 부적절하다.
이미 별표만 남은 값이나 `<KR_NAME>` 같은 완전 치환 태그는 새 가림의 정답으로
사용할 수 없어 입력 검증에서 거부한다.
개발용 파일을 `--development-input`으로 지정하면 최종 평가용과 같은 ID 또는
키 순서만 다른 동일 JSON 레코드가 있는지 검사하고 중단한다. 값만 다른 같은
고객·템플릿은 자동으로 찾지 못하므로 분할 단계에서 별도로 확인해야 한다.

```jsonl
{"id":"sample-001","record":{"name":"홍길동","phoneNumber":"010-1234-5678","order":"AB-123"},"targets":[{"path":["name"],"type":"KR_NAME"},{"path":["phoneNumber"],"type":"PHONE_NUMBER"}],"negative_paths":[["order"]]}
{"id":"sample-002","record":{"note":"접수"},"targets":[]}
```

`negative_paths`는 가리면 안 되는 값, `ignore_paths`는 주석 범위 밖이라 오탐에서
제외할 값을 선택적으로 지정한다. 세 목록은 겹칠 수 없다. `ignore_paths`가 많으면
오탐률이 왜곡되므로 지원 범위 밖 값에만 사용한다. 개인정보가 객체 **키**에 있는
경우는 아직 정답 경로로 지정할 수 없다. 보고서의 `changed_keys_out_of_scope`에
키 변경 횟수를 별도로 남기며, 이 값이 있으면 수동으로 확인해야 한다. 텍스트
필드의 일부만 개인정보라면 이 도구의 정답으로 넣지 말고 문자열 문서 평가의
문자 구간 주석을 사용한다.

```bash
.venv/bin/python benchmarks/structured_corpus_benchmark.py \
  --input private-eval/structured-holdout.jsonl --validate-only \
  --development-input private-eval/structured-development.jsonl \
  --entities KR_NAME PHONE_NUMBER EMAIL_ADDRESS \
  --output private-eval/structured-validation.json
.venv/bin/python benchmarks/structured_corpus_benchmark.py \
  --input private-eval/structured-holdout.jsonl \
  --development-input private-eval/structured-development.jsonl \
  --entities KR_NAME PHONE_NUMBER EMAIL_ADDRESS \
  --baseline-ref 30bf0b0 --output private-eval/structured-report.json
```

개발용 파일이 없으면 두 명령에서 `--development-input` 줄을 빼면 된다.

기본 기준 `30bf0b0`은 구조화 `mask_json()`이 처음 포함된 커밋이다. 그 전
버전에는 같은 API가 없어 비교할 수 없다. 이름·주소 NER이 필요한 경우에는
`--entities`에 `KR_NAME` 또는 `KR_ADDRESS`를 포함하고 `--ner`를 붙인다.
모델은 미리 로컬에 내려받아야 한다. 기준과 현재 프로세스는 같은 입력 복사본,
의존성, 고정 모델 revision을 사용한다.

`exact_type_fields`는 태그 출력이 정확히 `<TYPE>`인 정답 필드 수이고,
`fully_masked_fields`는 별표 출력이 원래 값 전체를 가린 필드 수다. 다른 유형으로
완전히 가린 값은 전자에서는 실패하지만 후자에서는 성공이다.
`changed_nontarget_fields`는 정답·제외 범위 밖의 값이 바뀐 필드 수이며,
`negative_records_with_changed_field`는 정답이 없는 레코드에서 하나라도 값이
바뀐 레코드 수다. 실제 원문이나 문서 ID, 필드 경로는 보고서에 저장하지 않는다.
이 수치가 실무 성능을 뜻하려면 업무 원본에서 수집한 레코드의 완전한 주석과
개발·최종 평가 분리가 필요하다. Git 저장소에 원문을 넣지 않는다.
