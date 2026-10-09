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
```

CI는 위 검사를 Python 3.10~3.13에서 실행합니다. 평가 결과와 명령은
[평가 문서](evaluation.md)에 있습니다. 합성 회귀, 외부 합성 평가, 속도 측정을 구분합니다.

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
