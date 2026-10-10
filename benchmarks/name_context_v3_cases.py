"""Freeze v3 synthetic splits before training; all inspected v2 data is development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from name_context_v2_cases import NAMES as V2_NAMES
from name_context_v2_cases import particles, render

ROOT = Path(__file__).parent
OUTPUT = ROOT / "data/name_context_v3.jsonl"
NAMES = {
    "train": tuple(dict.fromkeys(name for names in V2_NAMES.values() for name in names)),
    "validation": (
        "탁서린", "팽이솔", "갈도윤", "남궁하온", "황보서진", "샘", "담빛", "환희",
        "Lina Trial", "Omar Trial", "이든·가온", "도",
    ),
    "evaluation": (
        "설유담", "봉라온", "편지후", "제갈다빈", "선우채운", "독고세림", "구양서하",
        "건", "록", "채", "섬", "평온", "고요", "정성", "정직", "겸손",
        "Nora Fiction", "Ilan Fiction", "루아·다솔", "Ame\u0301lie Trial",
    ),
}
TEMPLATES = {
    "train": (
        "사회자 {a}{a_topic} 연주자 {b} 씨를 무대로 불렀다.",
        "물품을 찾아온 {a}{a_subject} 수령증에 서명했다.",
        "독자인 {a} 씨가 사서 {b} 씨에게 책을 반납했다.",
        "동행인 {a} 씨와 접수자인 {b} 씨가 함께 기다렸다.",
        "오늘 진행을 맡은 사람은 {a}이고 기록자는 {b}이다.",
        "내려받은 경로 보관/{a}_위임장.pdf를 확인했다.",
        "메일에 {a}_수료증.pdf와 {b}_지원서.pdf가 들어 있었다.",
        "정리 담당으로 {a}, {b} 두 사람이 지정되었다.",
        "신분을 확인하자 본인의 성함이 {a}{a_quotative} 답했다.",
        "작업자인 {a}{a_subject} 책임자 {b} 씨에게 보고했다.",
        "발표 자료에 적힌 {a}{a_topic} 발표자의 성명이다.",
        "명찰에 쓰인 {a}{a_topic} 이 자리에 참석한 사람의 이름이다.",
    ),
    "validation": (
        "전시를 설명하던 안내원 {a}{a_subject} 관람객 {b} 씨를 맞았다.",
        "심사를 마친 심사위원 {a}{a_topic} 참가자 {b} 씨에게 결과를 알렸다.",
        "축사를 낭독한 이는 {a}이며 통역을 담당한 이는 {b}이다.",
        "검토 목록의 파일 신청자료/{a}_확인서.pdf를 읽었다.",
        "배지를 보니 방문자의 성명은 {a}이었다.",
        "봉사자 명부에 {a}, {b} 두 사람을 추가했다.",
    ),
    "evaluation": (
        "분실물을 돌려받은 {a}{a_subject} 안내 데스크에 감사 인사를 남겼다.",
        "무대 뒤에서 대기하던 배우 {a}{a_topic} 연출가 {b} 씨를 불렀다.",
        "행사를 촬영한 이는 {a}이고 편집을 맡은 이는 {b}이다.",
        "보관함에서 꺼낸 인사자료/{a}_근무확인서.pdf를 읽어 보았다.",
        "도예가 {a} 씨와 수강생인 {b} 씨가 가마를 살폈다.",
        "이날의 안전 요원으로 {a}, {b} 두 사람이 배치되었다.",
        "접수원이 이름을 확인하자 방문자는 {a}{a_quotative} 답했다.",
        "전송 대기 중인 파일은 {a}_연구참여동의서.pdf였다.",
        "표창을 받은 {a} 씨에게 진행자 {b} 씨가 꽃을 건넸다.",
        "교환 학생인 {a}{a_topic} 지도 교사 {b} 씨에게 과제를 제출했다.",
        "다음 순서의 발표자는 {a}이며 질의자는 {b}이다.",
        "점검 기록의 작성자 {a}{a_genitive} 서명을 {b} 씨가 확인했다.",
    ),
}
WORDS = {
    "train": ("도움", "은혜", "성실", "열매", "수", "찬", "윤", "엘", "미소", "소망"),
    "validation": ("환희", "담빛", "햇무리", "설렘"),
    "evaluation": ("평온", "고요", "정성", "정직", "겸손", "물결", "산뜻함", "포근함"),
}
NEGATIVES = {
    "train": (
        "발표 자료에 적힌 {word}{topic} 사람 이름이 아닌 표제어였다.",
        "명찰 모양의 작품에 쓴 {word}{topic} 작가 이름이 아니라 주제이다.",
        "책 표지의 {word}{topic} 저자 성명이 아니라 제목이다.",
        "보고서의 {word}{topic} 업무 용어이며 인물을 가리키지 않는다.",
        "첨부 파일 {word}_개념설명.pdf는 용어 해설 자료이다.",
        "{word}라는 표현의 사전적 의미를 정리했다.",
    ),
    "validation": (
        "수업에서 다룬 {word}{topic} 사람을 부르는 말이 아닌 추상 명사다.",
        "문학 작품의 제목 {word}{object} 읽고 감상문을 썼다.",
        "첨부: {word}_용어사전.pdf (개념 해설)",
    ),
    "evaluation": (
        "게시판의 {word}{topic} 인물 소개가 아니라 이달의 표어다.",
        "해설자는 {word}{object} 감정을 나타내는 보통명사로 분류했다.",
        "저자란을 비워 둔 시집의 제목은 {word}였다.",
        "문서 {word}_어휘학습.pdf에는 낱말 풀이만 실려 있다.",
        "이번 토론은 {word}라는 개념의 뜻을 비교하는 자리였다.",
    ),
}


def row(identifier, split, context, text, expected, surface=None):
    return dict(
        id=identifier, split=split, context_id=context, text=text, expected=expected,
        surface=surface, track="prose",
        surfaces=sorted({text[e["start"] : e["end"]] for e in expected}),
    )


def build_cases():
    rows = []
    # Preserve original text/gold, while explicitly retiring every old split.
    for old in map(json.loads, (ROOT / "data/name_context_v2.jsonl").read_text().splitlines()):
        rows.append(row(
            "v3-development-" + old["id"], "train", "v3-development-" + old["context_id"],
            old["text"], old["expected"], old["surface"],
        ))
    for split, names in NAMES.items():
        for i, template in enumerate(TEMPLATES[split]):
            for j, name in enumerate(names):
                text, expected = render(template, name, names[(j + 5) % len(names)])
                rows.append(row(
                    f"v3-{split}-person-{i:02}-{j:03}", split,
                    f"v3-{split}-person-{i:02}", text, expected, name,
                ))
        for i, template in enumerate(NEGATIVES[split]):
            for j, word in enumerate(WORDS[split]):
                rows.append(row(
                    f"v3-{split}-negative-{i:02}-{j:02}", split,
                    f"v3-{split}-negative-{i:02}",
                    template.format(word=word, **particles(word)), [],
                ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in build_cases())
    if args.check:
        if OUTPUT.read_text(encoding="utf-8") != text:
            raise SystemExit("v3 corpus differs from authored generator")
    else:
        OUTPUT.write_text(text, encoding="utf-8")
    print(f"{len(build_cases())} synthetic v3 cases")


if __name__ == "__main__":
    main()
