# 재개 메모 (별명·핸들 파일럿, 잠정 상태로 저장)

현재 상태: 에이전트 2개의 1차 블라인드 판정까지 완료. **정답 미확정, 학습·제품 변경·예비셋 개봉 없음.** 상세는
`docs/name-handle-pilot.md`, 상태 표시는 `benchmarks/data/name_handle_pilot_v1.STATUS.md`.

남은 작업 (순서대로):
1. **A 미검증 판정 정리.** `reviewer_A_origin.json`의 `default_filled` 197행을 `review_unverified`로 분리하고, 일치 335건 중
   190건(모두 non_personal)을 검증된 합의에서 제외한 뒤 조정 큐와 감사 표본을 다시 만든다(현재 65건·34건은 잠정 작업량).
   A를 새 컨텍스트에서 항목별 근거와 함께 다시 실행할지, 해당 행만 재판정할지는 이때 정한다.
2. **곽두일 정책 확정.** `docs/name-handle-policy-proposal.md`의 4건. 확정 후 영향받는 모든 사례를 같은 지침으로 재판정한다.
3. **조정 및 감사.** 곽두일이 불일치·불충분·정책 미정 사례를 조정하고 합의 사례를 표본 감사한다. 최초 판정은 덮어쓰지 않는다.
이 세 단계가 끝나기 전에는 학습 사전 등록(초안 `name-p0-preregistration-amendment-3-draft.json`)을 확정하지 않는다.
