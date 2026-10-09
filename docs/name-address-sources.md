# 이름·주소 모델과 자료 출처

확인일: **2026-10-10**. 모델 선택의 근거와 재현에 필요한 버전을 기록합니다.
모델 저자의 점수는 ko-pii-guard의 측정 결과가 아닙니다. 이 프로젝트의 성능은
자체 실행한 평가 결과와 함께 확인해야 합니다. 이름(`KR_NAME`)과 주소(`KR_ADDRESS`)는
명시적으로 선택하는 엔티티이며, 기본 탐지 대상은 기존 식별번호 등 9종입니다.

## 선택한 로컬 NER 모델

| 항목 | 확인한 값 |
|---|---|
| 저장소 | [`FrameByFrame/korean-pii-e5-base`](https://huggingface.co/FrameByFrame/korean-pii-e5-base) |
| 프로필 | `e5` — `KoreanNER`의 기본 모델 |
| 고정 revision | `a308c54b4407819624a5661e31e162a269f39818` |
| 공개·최종 수정 | 2026-06-09·2026-06-10, Hugging Face API 메타데이터 기준 |
| 모델 형식 | 표준 `XLMRobertaForTokenClassification`, BIOES 37개 라벨 |
| 가중치 | `model.safetensors`, BF16 277,481,509개 파라미터 |
| 가중치 파일 크기 | 554,987,146바이트. 실행 시 전체 RAM 사용량과는 다름 |
| 입력 한도 | 토크나이저 512토큰, `max_position_embeddings=514`; 저자 학습 길이는 256토큰 |
| 가중치 라이선스 | 고정 revision 모델 카드와 별도 LICENSE 파일의 MIT 선언 |
| 기반 모델 | `intfloat/multilingual-e5-base`, MIT 선언 |
| 기본 NER 임계값 | `0.9`; 모델 신뢰도이며 보정된 정확도·확률 보장이 아님 |

고정 버전의 [모델 카드](https://huggingface.co/FrameByFrame/korean-pii-e5-base/blob/a308c54b4407819624a5661e31e162a269f39818/README.md),
[설정](https://huggingface.co/FrameByFrame/korean-pii-e5-base/blob/a308c54b4407819624a5661e31e162a269f39818/config.json),
[토크나이저 설정](https://huggingface.co/FrameByFrame/korean-pii-e5-base/blob/a308c54b4407819624a5661e31e162a269f39818/tokenizer_config.json),
[LICENSE](https://huggingface.co/FrameByFrame/korean-pii-e5-base/blob/a308c54b4407819624a5661e31e162a269f39818/LICENSE)를
확인했습니다. 미세조정 모델의 LICENSE는 저작권자를 Vijayachandran Mariappan으로 명시하며,
[기반 모델](https://huggingface.co/intfloat/multilingual-e5-base)의 MIT 라이선스와
학습 자료 KDPII의 CC BY 4.0도 명시합니다. 기반 모델의 허가만으로 미세조정 가중치의
허가를 추정한 것은 아닙니다.

ko-pii-guard 코드의 Apache-2.0과 선택적으로 내려받는 모델의 MIT는 각각 적용됩니다.
모델의 MIT 표기가 학습 원문이나 모든 의존성의 라이선스를 MIT로 바꾸지는 않습니다.
가중치를 재배포할 때도 해당 저작권·허가 고지와 기반 모델의 출처를 보존해야 합니다.

| 모델 라벨 | 설정 ID | ko-pii-guard 엔티티 |
|---|---|---|
| `B/I/E/S-private_person` | 1~4 | `KR_NAME` |
| `B/I/E/S-private_address` | 17~20 | `KR_ADDRESS` |
| `O` | 0 | 결과 없음 |
| 그 밖의 7개 범주 | 5~16, 21~36 | 이름·주소 어댑터에서 사용하지 않음 |

선택 이유는 한국어 문맥을 학습한 이름·주소 라벨, 명시된 MIT 라이선스와 개발용 문장에서
확인한 조사 경계 처리입니다. 같은 문장에 대한 KcELECTRA의 원시 예측과 비교했을 때 E5가
이름 본체와 조사를 더 정확하게 구분했습니다. 이는 개발 중의 모델 선택 근거이며 전국민
이름이나 독립적인 실제 서비스 성능을 대표하지 않습니다. 현재 성능 수치는 변경된 평가
자료로 다시 실행한 보고서를 기준으로 합니다.

모델 카드는 KDPII 대화 자료, KLUE-NER 뉴스의 인명 span, LLM이 만든 여러 문서 분야의
문장과 합성 PII를 섞어 학습했다고 설명합니다. KDPII 시험 자료, KLUE 검증 자료,
보험·정부 문서 분야를 학습에서 분리했다고 보고하지만, 이 프로젝트에서 원문·학습 로그·
분할 독립성을 감사하지 않았습니다. 학습 자료 전체의 이용 조건도 별도로 검증하지
않았으며, 가중치의 MIT 선언으로 학습 원문 재배포 허가까지 주장하지 않습니다.
이 패키지는 해당 학습 원문을 포함하지 않습니다.

저자의 공개 F1은 조사·공백을 제거하는 후처리를 포함합니다. 이 프로젝트는 이름 자체를
잘라낼 수 있는 저자의 일괄 조사 제거 함수를 사용하지 않으며 BIOES 라벨과 원문 오프셋으로
span을 복원합니다. 따라서 저자의 점수를 이 구현의 성능으로 가져올 수 없습니다.
저자도 합성 문서의 보류 분야가 학습과 생성기·PII 분포를 공유해 성능이 낙관적이라고
명시합니다. 같은 철자의 보통명사와 이름, 조사로 끝나는 이름, 복성, 희귀 이름, 긴 문장의
경계, 실제 음성 문장을 따로 평가해야 합니다.

512토큰 한도는 문자 수가 아닙니다. 긴 문장은 겹치는 토큰 구간으로 나누고 원문 오프셋을
복원해야 합니다. CPU 실행 가능성과 초당 처리량은 별개이며, 위 파일 크기로 지연시간을
추정하지 않습니다. 모델을 미리 내려받은 뒤 로컬에서 추론하면 분석 문장을 외부 추론
서비스에 보낼 필요가 없습니다.

## 비교 검토한 모델

| 모델 | 라이선스·규모 | 선택 시 주의점 |
|---|---|---|
| [`kiyuyeon/pii-ner-kcelectra`](https://huggingface.co/kiyuyeon/pii-ner-kcelectra) | MIT 선언, 127M, F32 가중치 약 509MB | `kcelectra` 프로필로 선택 가능. BIO의 `NAME`→`KR_NAME`, `ADDR`→`KR_ADDRESS`. 원시 예측의 조사 경계에 주의 |
| [`urchade/gliner_multi-v2.1`](https://huggingface.co/urchade/gliner_multi-v2.1) | Apache-2.0, 현재 safetensors 메타데이터 289M·F32 약 1.16GB | 유연한 라벨을 받는 다국어 모델. 카드에서 한국어 PII 이름·주소의 독립 검증 점수는 확인하지 못함 |
| [`Leo97/KoELECTRA-small-v3-modu-ner`](https://huggingface.co/Leo97/KoELECTRA-small-v3-modu-ner) | 약 14M | 확인한 모델 카드에 미세조정 가중치 라이선스 선언이 없어 기본 모델로 선택하지 않음 |
| [`vitus9988/klue-roberta-small-ner-identified`](https://huggingface.co/vitus9988/klue-roberta-small-ner-identified) | 약 68M | 확인한 모델 카드에 미세조정 가중치 라이선스 선언이 없어 기본 모델로 선택하지 않음 |

KcELECTRA 프로필 revision은 `cf4b4d54fe0c4168c2796a186ea4cb238b6557ca`입니다. 해당
[모델 카드](https://huggingface.co/kiyuyeon/pii-ner-kcelectra/blob/cf4b4d54fe0c4168c2796a186ea4cb238b6557ca/README.md)는
MIT를 선언하지만 별도 LICENSE 파일은 확인하지 못했습니다.
[기반 모델 카드](https://huggingface.co/beomi/KcELECTRA-base-v2022/blob/87285578b4686b6d190abd40e947a2a416add6de/README.md)에도
MIT와 출처 표기가 명시돼 있습니다. 저자는 38종 템플릿 합성 문장에 커뮤니티·뉴스 자료를
추가했다고 설명하며, 공개된 이름 검출 점수는 정확한 span F1이 아닌 겹침 재현율입니다.
크롤링 원문 이용 조건과 훈련·평가 분할은 여기서 별도로 감사하지 않았습니다.

검토한 GLiNER revision은 `443d26d654e0324125a96bebd8e796c14ff2efe6`입니다. E5와
KcELECTRA를 제외한 위 후보는 모델 카드와 파일만 검토했으며 같은 자료로 실행해 비교한
결과가 아닙니다. `gliner_multi_pii-v1`의
모델 카드가 열거하는 언어에는 한국어가 없으므로 이름에 `multi`가 있다는 이유만으로
한국어 성능을 보장하지 않습니다.

## 주소 표기와 행정구역 사전의 공식 출처

| 자료 | 확인 내용 | 제공처가 표시한 이용 범위 |
|---|---|---|
| [행정안전부 행정표준코드 법정동코드](https://www.data.go.kr/data/15077871/openapi.do) | 시도·시군구·읍면동·리 코드와 지역명, 상위 코드·생성일. API 인증키 필요 | 이용허락범위 제한 없음 |
| [행정표준코드관리시스템 법정동코드 조회](https://www.code.go.kr/stdcode/regCodeL.do) | 원천 코드 조회와 행정구역 변경 확인 | 위 공식 공개 데이터와 대응하는 원천 시스템 |
| [행정안전부 도로명주소 주소DB](https://www.data.go.kr/data/15050417/fileData.do) | 도로명주소 표기법, 전체분·변동분 제공 안내 | 이용허락범위 제한 없음 |

주소DB 안내는 시도, 시군구, 읍면, 도로명, 건물번호, 상세주소(동·층·호), 참고항목의
순서를 설명합니다. 이 표기법은 주소 후보 경계의 근거이며 주소의 실재 여부 검증과는
다릅니다. `서울`이나 `강남구`처럼 지역만 언급한 문장과 구체적인 배송·거주 주소는
평가에서 구분해야 합니다.

전국 사전을 배포한다면 공식 원천의 다운로드 날짜, 데이터 기준일, SHA-256, 출처,
이용 조건과 폐지 코드 처리 방법을 함께 기록해야 합니다. 행정구역 명칭은 변경되므로
현행 목록과 과거 주소의 명칭을 구분해 유지합니다. 위 링크를 인용했다고 해서 전국
주소DB나 모든 시군구 사전이 이 패키지에 포함된 것은 아닙니다. 특정 주소 문장을 외부
주소검색 API에 전송하는 대신 공개 사전 파일을 미리 확보해 로컬에서 사용할 수 있습니다.

## 중국어 NER에서 참고할 방법과 한국어 적용 범위

중국어 연구에서 참고할 수 있는 것은 사전·경계·문맥을 결합하는 학습 방식이다.
중국어 모델 가중치나 분절기를 그대로 한국어에 적용하면 같은 성능이 나온다는 근거는 아니다.

- [SoftLexicon (ACL 2020)](https://aclanthology.org/2020.acl-main.528/)은 사전에서
  일치한 후보를 문자 표현에 추가한다. 사전 일치를 곧 인명이라는 확정 규칙으로 쓰지 않는다.
- [FLAT (ACL 2020)](https://aclanthology.org/2020.acl-main.611/)은 겹치는 문자·단어
  후보와 상대 위치를 함께 학습한다. 한국어 형태소·조사 후보를 넣는 것은 별도 실험 과제다.
- [NerCo (IJCAI 2023)](https://www.ijcai.org/proceedings/2023/587)는 의미 범주에 따른
  대조 학습과 후속 NER 학습을 사용한다. 한국어 업무 상용구 오탐 감소가 검증된 결과는 아니다.
- [Name Regularity Bias (TACL 2021)](https://aclanthology.org/2021.tacl-1.36/)는
  이름 표면형에 의존하는 편향을 진단하고 문맥 신호를 더 사용하도록 학습하는 방법을 다룬다.
- [한국어 언어 특성을 이용한 NER](https://arxiv.org/abs/2305.06330)는 조사 등 기능
  형태소와 이름 본체의 경계를 구분하는 주석 형식을 연구한다. 형태소 분석 출력으로 이름
  끝글자를 무조건 제거하라는 근거가 아니다.

후속 실험에서는 같은 표면형이 인명/일반명사로 쓰인 대조 문장, 업무 역할명, 조사로 끝나는
이름을 포함하고 **이름과 문장 틀을 함께 분리한** 미사용 평가 자료를 확보하는 것이 우선이다.
현재의 합성 회귀 세트를 학습에도 사용한다면 그 점수를 독립 평가로 내세울 수 없다.

이 논문들은 참고 문헌이다. 현재 패키지는 사전 lattice, 대조 학습, adversarial training을
구현하거나 새 모델을 재학습하지 않았다. 현재 구현은 고정 공개 모델, 문맥별 보수적 규칙,
원문 구간 검증과 프로필별 회귀 평가다. 한국어에서의 개선 효과는 실제 비교 실험 전에는
주장하지 않는다.
