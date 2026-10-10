# 다운로드한 논문

[name-multisource-v2.md의 "최근 연구 확인"](../name-multisource-v2.md#최근-연구-확인)에서
인용하는 논문 중 실제로 로컬에 보관한 것들이다. 링크만 걸고 원문을 저장하지
않았던 이전 상태(GLiNER2-PII, Meddies-PII v2, Thunder-DeID)와 구분하기 위해,
2026-10-10에 새로 찾은 것부터는 PDF/HTML 원문을 이 폴더에 받아 둔다.
저장된 날짜와 검색 질의는 각 파일 설명에 적는다. arXiv/PMC 원문의 저작권은
각 저자/발행처에 있으며, 이 저장소는 연구 참고용으로만 보관한다.

| 파일 | 출처 | 날짜 | 비고 |
|---|---|---|---|
| `redact-2606.19881.pdf` | [arXiv:2606.19881](https://arxiv.org/abs/2606.19881) | 제출 2026-06-18 | 25개 언어(한국어 포함) PII 탐지기 비교. Presidio가 모든 언어에서 최하위, 이름 구성요소 유형은 전 탐지기에서 약함 |
| `tides-2608.01724.pdf` | [arXiv:2608.01724](https://arxiv.org/abs/2608.01724), COLM 2026 | 2026-08 제출 | 한국어 7개 팀 회의록 비식별화에서 정규식 기반 접근이 91% 오탐률을 보여 LLM(Qwen3-30B) 기반으로 교체; 교체 후에도 375건 수동 보정 필요 |
| `korean-emr-deid.html` | [PMC13354230](https://pmc.ncbi.nlm.nih.gov/articles/PMC13354230/) (npj Health Syst, DOI 10.1038/s44401-025-00036-1) | 2025-09-04 | 한국어 퇴원요약지 비식별화, KLUE BERT + 증강. PER(인명) 엔티티 P 0.90 / R 0.82 / F1 0.91 |

2026-10-10 기준 검색 질의: "Korean PII de-identification NER 2026 arxiv person
name redaction", "Korean named entity recognition privacy masking LLM 2025
2026 arxiv". 같은 검색에서 나왔지만 한국어·인명 탐지와 직접 관련이 없어 받지
않은 것: REDACT와 겹치는 일반 다국어 벤치마크(PIIBench), 영어 중심 PII
마스킹 모델 점검(arXiv:2504.12308), 텍스트 비식별화 일반 서베이
(arXiv:2508.21587). `korean-emr-deid.html`은 2025-09-04로, 이 저장소가 기준으로
잡은 "최근 1년"(2025-10-10~2026-10-10) 바로 바깥이지만 한국어 인명
비식별화라는 주제가 직접 겹쳐 함께 보관한다.
