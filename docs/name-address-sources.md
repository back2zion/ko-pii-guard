# 이름·주소 모델과 자료 출처

확인일: **2026-10-10**. 모델 선택의 근거와 재현에 필요한 버전을 기록합니다.
모델 저자의 점수는 ko-pii-guard의 측정 결과가 아닙니다. 이 프로젝트의 성능은
자체 실행한 평가 결과와 함께 확인해야 합니다. 이름(`KR_NAME`)과 주소(`KR_ADDRESS`)는
명시적으로 선택하는 **개발 중인 실험적 엔티티**이며, 기본 탐지 대상은 기존 식별번호 등
9종입니다. 소규모 개발 결과는 [기술 회귀 평가](name-address-evaluation.md)에 기록합니다.
평가 기준은 [이름 span 주석 계약](name-span-contract.md)과
[문맥 평가 계획](name-context-evaluation.md)을 따릅니다.

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

중국어 연구에서 참고할 수 있는 것은 사전·경계·문맥을 결합하는 학습 방식입니다.
중국어 모델 가중치나 분절기를 그대로 한국어에 적용하면 같은 성능이 나온다는 근거는
아닙니다. 특히 중국어 단어 분절과 한국어 이름·조사 경계는 같은 문제가 아닙니다.

**SoftLexicon**은 사전에서 일치한 단어 후보를 문자별 B/M/E/S 집합으로 나누고, 단어
임베딩을 모아 문자 표현에 결합하는 모델 구조입니다. 사전 일치를 인명이라는 확정 규칙으로
사용하지 않습니다. 현재 E5에 이 방식을 도입하려면 표현 계층을 바꾸고 재학습해야 합니다.
사전 일치 여부로 이미 나온 NER 점수만 더하거나 빼는 휴리스틱은 별개의 **점수 보정 실험**이며
SoftLexicon 구현이라고 부를 수 없습니다.

**Name Regularity Bias(NRB)** 연구는 이름 표면형에 의존하는 편향을 진단하고,
학습 가능한 adversarial noise 등을 사용해 문맥 의존을 높이는 훈련 방법을 제안합니다.
이 프로젝트가 먼저 참고하는 부분은 같은 표면형을 다른 문맥에 넣는 대조 진단입니다.
이를 한국어 업무 문장으로 구성하는 것은 NRB에서 착안한 평가이며, 원래 NRB 데이터셋이나
논문 결과를 재현했다는 뜻이 아닙니다. 해당 adversarial training은 구현하지 않았습니다.

후속 작업의 순서는 다음과 같습니다.

1. [주석 계약](name-span-contract.md)에 따라 인명/일반명사 대조 문장, 업무 역할명,
   조사로 끝나는 이름을 포함한 challenge set을 확정하고 동결합니다. 이름과 문장 틀을
   함께 분리하며, 모델 출력에 맞춰 정답을 고치지 않습니다.
2. 동결한 자료에서 고정 공개 모델과 현재 규칙을 기준선으로 평가하고,
   오탐·미탐·경계 오류를 분리합니다. 기존의 작은 개발 세트는 기술 회귀 기록으로 유지합니다.
3. [문맥 평가 계획](name-context-evaluation.md)에 따라 점수 보정이 없는 기준선과
   사전·문맥별 보정을 하나씩 비교합니다. 보정 설정은 별도의 개발 자료에서 정하고,
   동결한 평가 자료에서 오탐 감소와 누락·마스킹 범위의 손실을 함께 확인합니다.
4. 남은 오류와 비교 결과가 구조 변경의 필요성을 뒷받침할 때만 SoftLexicon 등의
   재학습을 검토합니다. 재학습에는 별도의 학습 자료와 사용 권한, 평가 분리가 필요합니다.

현재 구현은 고정 공개 모델, 문맥별 보수적 규칙, 원문 구간 검증과 프로필별 회귀 평가입니다.
사전 lattice, 논문의 대조 학습이나 adversarial training을 구현하거나 새 모델을 재학습하지
않았습니다. 한국어에서의 개선 효과는 실제 비교 실험 전에는 주장하지 않습니다.

### 핵심 참고 문헌

- Ruotian Ma, Minlong Peng, Qi Zhang, Zhongyu Wei, and Xuanjing Huang. 2020.
  **Simplify the Usage of Lexicon in Chinese NER.** Proceedings of the 58th Annual
  Meeting of the Association for Computational Linguistics, pp. 5951–5960.
  [ACL 2020](https://aclanthology.org/2020.acl-main.528/),
  [arXiv:1908.05969](https://arxiv.org/abs/1908.05969).
- Abbas Ghaddar, Philippe Langlais, Ahmad Rashid, and Mehdi Rezagholizadeh. 2021.
  **Context-aware Adversarial Training for Name Regularity Bias in Named Entity
  Recognition.** Transactions of the Association for Computational Linguistics,
  9:586–604. [TACL 2021](https://aclanthology.org/2021.tacl-1.36/),
  [arXiv:2107.11610](https://arxiv.org/abs/2107.11610).

추가로 [FLAT (ACL 2020)](https://aclanthology.org/2020.acl-main.611/)의 겹치는 문자·단어
후보 표현, [NerCo (IJCAI 2023)](https://www.ijcai.org/proceedings/2023/587)의 대조 학습,
[한국어 언어 특성을 이용한 NER](https://arxiv.org/abs/2305.06330)의 기능 형태소와 이름
경계 주석을 검토했습니다. 이들도 참고 문헌이며 현재 패키지에 구현한 방법이 아닙니다.
