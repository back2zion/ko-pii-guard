# 개발·평가 안내

이 저장소는 한국어 텍스트의 구조화된 개인정보를 탐지하고 마스킹하는 라이브러리입니다.
현재 작업 트리의 새 기능은 아직 PyPI에 배포되지 않았습니다. 이름·주소는 아래 범위에서
지원하며 건강보험 번호는 지원하지 않습니다.

## 설치와 사용

```bash
uv sync --frozen --extra dev --extra guardrails
uv run python -c "from ko_pii_guard import KoreanPIIGuard; print(KoreanPIIGuard().mask('입금은 농협 302-1234-5678-91 로 부탁드립니다'))"
```

출력은 `입금은 농협 <KR_ACCOUNT> 로 부탁드립니다`입니다. 실제 계좌 존재 여부나 소유자를
확인하는 기능은 아닙니다. 서비스 시작 시 `KoreanPIIGuard()`를 한 번 만들고 재사용하세요.

```python
from ko_pii_guard import KoreanPIIGuard

guard = KoreanPIIGuard()
text = "문의는 user@example.co.kr로 보내세요"
findings = guard.analyze(text)
assert all(text[f.start:f.end] == f.text for f in findings)
assert guard.mask(text) == "문의는 <EMAIL_ADDRESS>로 보내세요"
```

`start`/`end`는 원문 Python 문자열의 문자 인덱스이며 `end`는 포함하지 않습니다.
UTF-8 바이트나 JavaScript UTF-16 인덱스가 아닙니다. 전각 숫자·하이픈·숫자 사이의 일부
제로폭 문자도 처리하며, 결과와 마스킹은 원문 위치를 유지합니다.
`normalize_unicode=False`로 정규화를 끌 수 있습니다.

`tag`는 엔티티 태그, `stars`는 탐지 구간 전체 별표, `partial`은 정해진 앞부분을
남깁니다. 부분 마스킹은 주민번호 생년월일 등 일부 내용을 공개하므로 용도에 맞게 고르세요.
인접한 두 값은 각각 치환하고 그 사이 공백·구두점은 보존합니다.

## 탐지 정책

- 이름·주소는 `entities`에 `KR_NAME`, `KR_ADDRESS`를 명시해야 켜집니다.
  이름 규칙은 명시적인 성명·예금주 등의 필드에서 탐지합니다. 이름 길이나 성씨 사전만으로 일반
  문장의 이름을 추측하지 않습니다. 자유 문장은 `KoreanNER`를 연결해 문맥으로 판정합니다.
  이름 본체와 조사의 경계는 [API 구간 계약](name-span-contract.md)을 따릅니다.
  탐지와 마스킹의 좌표 보존 검증을 학습 모델의 인식 정확도로 해석하지 않습니다.
- 주소는 시도·행정구역 또는 명시적 주소 라벨과 도로명/지번·번호를 요구합니다. 주소 사전의
  전수 매칭이나 실제 배송 가능성 검증이 아닙니다. 단순 지명은 주소로 취급하지 않습니다.
- 주민번호·사업자번호·카드·전화 등 기존 식별번호가 계좌 후보와 겹치면 기존 유형을 우선합니다.
- 계좌는 10~14자리 후보를 검사합니다. 숫자만 있거나 4그룹인 경우 같은 문장/필드의 앞 20자
  안에 계좌 문맥이 필요합니다. 4그룹은 [문헌으로 확인한 형태](account-format-sources.md)만 추가했습니다.
- `우리 상품`, `하나 남은` 같은 표현은 은행 라벨로 취급하지 않습니다. `우리 3333…`처럼
  라벨 자리에 홀로 쓰인 경우는 허용하므로 모호성이 완전히 사라지는 것은 아닙니다.
- 한국어 조사는 이메일·카드번호 뒤에 붙을 수 있지만 영문 코드 속 숫자 조각은 제외합니다.
- 불가능한 생년월일은 문맥 없는 주민번호 후보에서 제외합니다. 명시적으로 주민번호라고
  적힌 오타는 마스킹 대상으로 유지합니다. 2020년 이후 번호에 체크섬을 새로 강제하지 않습니다.
- `score`는 규칙 점수이며 확률이 아닙니다. 임계값을 올리면 미탐이 증가할 수 있습니다.
- 정규식 제한 시간을 초과하면 `TimeoutError`가 전달됩니다. 호출 서비스는 이때 원문을
  외부로 전달하지 않고 실패 경로로 처리해야 합니다.

## 검증과 기여

```bash
OTEL_SDK_DISABLED=true uv run pytest -q
uv run ruff check .
uv run python benchmarks/synthetic_benchmark.py --check
uv run python benchmarks/robustness_benchmark.py --check
uv run python benchmarks/name_address_benchmark.py --check
uv run python benchmarks/name_address_benchmark.py --data benchmarks/data/name_field_boundaries.jsonl --check --check-negatives
```

CI는 위 검사를 Python 3.10~3.13에서 실행합니다. 평가 결과와 명령은
[평가 문서](evaluation.md)에 있습니다. 합성 회귀, 외부 합성 평가, 속도 측정을 구분합니다.

이름 필드의 가운데점·호칭 공백 회귀는 필드 양성 16건·음성 7건과 별도 자유 문장
양성 1건을 포함합니다. 원본 v1은 자유 문장 양성을 음성으로 잘못 주석했으므로
현행 자료에서는 고쳤고 [주석 수정 기록](../benchmarks/data/README.md)을 남겼습니다.
2026-10-10 v1 변경 전/후 동일 자료에서 필드 정확 구간은 3/16→16/16,
전체 마스킹은 7/16→16/16, 음성 오탐은 0/8→0/8이었습니다.
[변경 전 보고서](../benchmarks/results/name-field-boundaries-before.json)와
[변경 후 보고서](../benchmarks/results/name-field-boundaries-after.json)에 소스·자료 해시가
있습니다. 수정에 사용한 개발 회귀이며 자유 문장 NER의 정확도 개선으로 해석하지 않습니다.
기존 이름 문맥 동결 자료와 보고서는 그대로 보존합니다.

새 유형이나 은행 형태를 추가할 때는 다음을 함께 제출해 주세요.

1. 공식 출처 URL·확인일·정확한 적용 범위. 상품 한 가지의 형식을 은행 전체 규격으로 넓히지 않습니다.
2. 사람이 정한 원문 정답 구간과 엔티티. 현재 탐지기의 출력을 정답으로 복사하지 않습니다.
3. 정상 사례뿐 아니라 날짜·전화·주문번호·잘린 숫자·문맥 없는 값의 반례.
4. 기존 형식과의 충돌, 마스킹 후 민감한 구간이 남지 않는지, 오탐과 미탐의 변화.
5. 실행 명령·고정 시드·데이터 해시가 있는 결과. 확인하지 않은 수치를 문서에 적지 않습니다.

버그 제보에는 실제 개인정보 대신 형태와 문맥을 보존한 합성 재현 사례를 사용해 주세요.
이름·주소의 [출처](name-address-sources.md)와 [평가 범위](name-address-evaluation.md)를 확인해 주세요.

## 문맥 모델 설치와 재현

```bash
uv sync --frozen --extra dev --extra guardrails --extra ner
uv run python -m ko_pii_guard.ner --download
KO_PII_TEST_NER=1 uv run pytest -q tests/test_ner.py
uv run python benchmarks/name_address_benchmark.py --ner --check --check-negatives --numeric-negatives --output benchmarks/results/name-address.json
uv run python benchmarks/name_address_benchmark.py --data benchmarks/data/business_korean.jsonl --ner --check-negatives --output benchmarks/results/business-korean.json
```

모델 다운로드는 설치 단계에서 명시적으로 실행합니다. `KoreanNER.from_pretrained()`는
기본적으로 로컬 캐시만 읽습니다. 문장은 외부 추론 API로 전송하지 않습니다. 운영 환경에서는
동일한 고정 revision의 파일을 미리 준비하고 모델을 한 번 로드해 재사용하세요. 기본 프로필 CI는 모델을 설치하지 않습니다. 별도 Python 3.12 NER 작업은 고정 모델을
내려받아 캐시하고 실제 통합 테스트·이름주소·업무 상용구 오탐 검사를 수행합니다.

이 저장소의 `uv.lock`은 선택적 `ner` 의존성에 CPU PyTorch를 사용합니다.
잠긴 macOS PyTorch wheel은 macOS 14 이상·ARM64용이며 Intel Mac용 wheel은 없습니다.
이 제약은 선택적 NER 환경에 적용됩니다. 기본 프로필의 Python 3.10~3.13 CI는
Linux에서 실행하므로 다른 운영체제까지 검증한 호환성 수치로 해석하지 마세요.

이름·주소 포함 프로필은 `KoreanPIIGuard(entities=SUPPORTED_ENTITIES,
ner=KoreanNER.from_pretrained())`로 만든다. `SUPPORTED_ENTITIES`는 `ko_pii_guard`에서
import한다. `ner`만 전달하면 기본 식별번호 범위가 그대로 유지된다.
파일명 안의 실제 이름도 탐지한다. 경로 보존 같은 치환 정책은 `should_mask` 콜백에서
별도로 지정하며, 탐지되지 않은 것으로 처리하지 않는다.

이름 개선은 [문맥 대조 평가](name-context-evaluation.md)의 데이터·설정을 먼저 동결하고
현재 모델을 측정하는 순서로 진행한다. 사전 점수 가산은 벤치마크에만 있는 비교 실험이며
제품 설정이나 재학습 모델이 아니다. 작은 회귀 세트의 점수를 README 정확도로 옮기지 않는다.

v2 문자 판정기는 `KoreanNER.from_pretrained(name_context_path="artifacts/name-context-v2")`로
명시적으로 연결한다. 알려진 v1 미탐 21건과 회귀 7건을 해결했고 파일명·다중 인물·긴 문장
통합 검사를 포함한다. 새 평가의 성능과 남은 오류는 [v2 평가](name-context-v2-evaluation.md)를
따른다. 기존 v1 분할은 v2의 학습/개발 자료이며 새 평가 성능에 합산하지 않는다.

```bash
uv run --frozen python benchmarks/name_context_v2_cases.py --check
uv run --frozen python benchmarks/name_context_v2_benchmark.py --output /tmp/name-context-v2-acceptance.json --check
```

v3는 기존 v2 오류 26문장을 개별 테스트로 고정한 후 학습한 별도 선택 체크포인트다.
`name_context_path="artifacts/name-context-v3"`로 연결한다. v2 전체 자료는 v3에서
개발 자료이며, 별도로 고정한 새 280문장의 성능과 한계는
[v3 재현·평가 기록](name-context-v3-evaluation.md)을 따른다.

```bash
uv run --frozen python benchmarks/name_context_v3_cases.py --check
KO_PII_TEST_NER=1 KO_PII_TEST_NAME_CONTEXT=1 KO_PII_TEST_NAME_CONTEXT_PATH=artifacts/name-context-v3 uv run --frozen pytest -q
uv run --frozen python benchmarks/name_context_v3_benchmark.py --split development --output /tmp/name-context-v3-acceptance.json --check
```


v6 후보 판정기는 기존 v3 이름 가중치를 보존하며 비인물 후보를 추가 판정한다.
`name_context_path="artifacts/name-context-v6-filter-retry"`로 명시적으로 선택한다.
[평가와 실행 명령](name-context-v6-evaluation.md), [회귀 계약](name-context-nonregression.md)을
따른다. CI는 기존 이름 구간 손실과 새 오탐 구간을 각각 거부한다. 전체 선택 모델
통합 테스트 1,354개가 통과했으며, 새 평가의 기존 미탐 9개는 남아 있다.


v7은 `name_context_path="artifacts/name-context-v7"`로 명시적으로 선택한다. 기존 v6
미탐 9개를 별도 테스트로 고정해 모두 해결했고 전체 통합 테스트 1,366개가 통과했다.
기존 판정기는 보존하며 겹치지 않는 이름 후보만 추가한다. CI는 새 평가와 과거 자료의
기존 정답 손실·새 오탐을 구간별로 거부한다. [명령·결과·한계](name-context-v7-evaluation.md).


v9는 `name_context_path="artifacts/name-context-v9"`로 선택한다. v7의 남은 미탐
6개·오탐 4개를 모두 해결했고 전체 1,381개 테스트를 통과했다. 추가 학습으로
기존 정답을 잃은 v8은 채택하지 않는다. v9는 v7 보조 판정기를 그대로 두고
개선 판정기를 추가하며, CI는 과거 자료와 새 평가를 구간별로 비교한다.
[실패 재현·원인·명령·새 평가 한계](name-context-v9-evaluation.md).

v11은 `name_context_path="artifacts/name-context-v11"`로 선택한다. v9의 미탐
5개·오탐 2개를 재현 테스트로 고정해 모두 해결했다. 새 평가에서 기존 이름을 잃은
v10은 거부했으며 해당 구간도 별도 테스트로 고정했다. 전체 1,391개 테스트와
과거 3,049문장의 구간별 보존 검사를 통과했다. 별도 새 평가에는 FN23/FP17이
남는다. CI는 기존 정답 손실이나 새 오탐을 각각 거부한다.
[검사 명령·선택·캐시 재생성·한계](name-context-v11-evaluation.md).
