"""Deterministic paired contexts for the nickname/handle scope contrast (P0, fixed pre-result).

The same surface string appears once in a context that refers to a person's own nickname,
stage name or account handle (expected: protected) and once in a non-personal context such as a
project code name, product, team or shared/organization account (expected: not protected).
Strings are ordinary Korean words and Latin handles; no real person is intended.
Rules are fixed before any model is run on these sentences; changing them voids the result.
"""

from __future__ import annotations

import hashlib
import json

WORDS = ["은빛고래", "달무리", "푸른솔", "하늘별", "소나기", "별빛소년", "노을빛", "꼬마여우",
         "새벽안개", "바람언덕", "moonlight_77", "blue.fox", "sky_walker", "kimchi_lover",
         "night_owl", "coffee-bean"]
PERSONAL = [
    "제 닉네임은 '{w}'입니다.",
    "저는 온라인에서 {w}라는 활동명으로 지냅니다.",
    "게임 아이디는 {w}이고 개인 계정입니다.",
    "제 인스타그램 아이디가 {w}예요.",
    "친구들은 저를 {w}라고 부릅니다.",
    "블로그에서 제 필명은 {w}입니다.",
]
NON_PERSONAL = [
    "프로젝트 코드명은 '{w}'입니다.",
    "이번에 출시되는 서비스 이름은 {w}입니다.",
    "고객센터 공용 계정은 {w}이고 담당자 개인과 무관합니다.",
    "우리 팀 이름이 {w}로 정해졌습니다.",
    "행사 슬로건은 {w}로 확정되었습니다.",
    "신제품 모델명이 {w}입니다.",
]


def build():
    rows = []
    for t, (personal, other) in enumerate(zip(PERSONAL, NON_PERSONAL, strict=True)):
        for w in WORDS:
            for kind, template in (("personal", personal), ("non_personal", other)):
                text = template.format(w=w)
                start = text.index(w)
                number = WORDS.index(w)
                rows.append(dict(id=f"handle-{t}-{number}-{kind}", pair=f"{t}-{number}",
                                 kind=kind, text=text, span=[start, start + len(w)],
                                 expected_protected=kind == "personal"))
    return rows


def digest(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


if __name__ == "__main__":
    rows = build()
    print(len(rows), digest(rows))
