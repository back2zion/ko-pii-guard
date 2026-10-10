"""V4 contextual non-person contrasts; fresh evaluation is frozen before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from name_context_v3_cases import NAMES as OLD_NAMES
from name_context_v3_cases import WORDS as OLD_WORDS
from name_context_v3_cases import particles, render, row

ROOT = Path(__file__).parent
OUTPUT = ROOT / "data/name_context_v4.jsonl"
NAMES = {
    "train": tuple(dict.fromkeys(n for names in OLD_NAMES.values() for n in names)),
    "validation": ("변이루", "옥채아", "사공유하", "동방서린", "둘", "모", "자비", "정갈",
                   "Lena Model", "Sami Model"),
    "evaluation": ("견하람", "빙서율", "옹다인", "황목리안", "어금소유", "서문도하",
                   "욱", "벼", "잎", "숲", "자유", "온정", "연민", "생기",
                   "Tara Fixture", "Emil Fixture", "미르·새온", "Le\u0301a Fixture"),
}
WORDS = {
    "train": tuple(dict.fromkeys(w for words in OLD_WORDS.values() for w in words)),
    "validation": ("자비", "정갈", "상쾌함", "푸름", "잔향", "노을빛"),
    "evaluation": ("자유", "온정", "연민", "생기", "선선함", "산들바람", "눈부심", "적막",
                   "넉넉함", "잔잔함"),
}
POSITIVE = {
    "train": (
        "참석자 {a} 씨는 오늘의 표어에 대해 설명했다.",
        "도예가 {a} 씨와 수강생인 {b} 씨가 작품을 살폈다.",
        "원고 작성자 {a}{a_topic} 감정을 묘사했다.",
        "문서 {a}_참가신청서.pdf에는 신청자의 성명이 실려 있다.",
        "게시판의 인물 소개에는 {a} 씨의 성함이 적혀 있다.",
    ),
    "validation": (
        "지휘자 {a}{a_subject} 단원 {b} 씨에게 신호를 보냈다.",
        "파일 제출/{a}_서약서.pdf를 수신했다.",
        "이 문장을 낭독한 독자의 성함은 {a}이었다.",
        "봉투에는 발송인 {a} 씨와 수령인 {b} 씨의 이름이 적혀 있었다.",
    ),
    "evaluation": (
        "견학을 마친 {a}{a_subject} 인솔자인 {b} 씨에게 인사했다.",
        "봉사 활동을 이끈 이는 {a}이고 기록을 남긴 이는 {b}이다.",
        "발송할 서류의 경로는 등록/{a}_활동확인서.pdf였다.",
        "정원사 {a} 씨와 방문자인 {b} 씨가 나무를 살폈다.",
        "새 위원 명부에 {a}, {b} 두 사람을 등록했다.",
        "안내원이 성명을 묻자 손님은 {a}{a_quotative} 답했다.",
        "수상자의 이름 {a}{a_object} 사회자 {b} 씨가 불렀다.",
        "배달을 요청한 {a} 씨에게 상담원 {b} 씨가 확인 전화를 했다.",
    ),
}
NEGATIVE = {
    "train": (
        "게시판의 {word}{topic} 인물 소개가 아니라 이달의 표어다.",
        "해설자는 {word}{object} 감정을 나타내는 보통명사로 분류했다.",
        "문서 {word}_어휘학습.pdf에는 낱말 풀이만 실려 있다.",
        "어휘 사전에 {word}의 뜻이 실려 있다.",
        "첨부 문서 {word}_낱말풀이.txt는 단어의 의미를 설명한다.",
        "전시의 주제인 {word}{topic} 인물을 지칭하는 표현이 아니다.",
        "이 글의 {word}{topic} 저자의 성함이 아니라 추상적 개념이다.",
        "도서 제목 {word}{object} 목록에 추가했다.",
    ),
    "validation": (
        "전시장 벽의 {word}{topic} 작가 이름 대신 적어 놓은 작품 제목이다.",
        "단어장 {word}_뜻풀이.pdf를 수업 자료로 썼다.",
        "교재에서 {word}라는 추상 명사를 설명했다.",
    ),
    "evaluation": (
        "이번 캠페인의 구호 {word}{topic} 어느 사람의 성명도 아니다.",
        "수업 자료 중 어휘/{word}_의미정리.pdf는 사전 풀이를 담고 있다.",
        "언어학자는 {word}{object} 보통명사의 한 예로 들었다.",
        "도서관 검색 결과의 {word}{topic} 저자가 아닌 책 제목이다.",
        "추상 개념 {word}에 대한 해설을 노트에 옮겼다.",
        "전시실 안내판에는 주제어 {word}만 적혀 있고 인명은 없다.",
    ),
}
MIXED = {
    "train": '작품 제목은 "{word}"이며 작가는 {a} 씨다.',
    "validation": '시집 "{word}"를 낭독한 이는 {a} 씨다.',
    "evaluation": '책의 표제는 "{word}"이고 저자는 {a} 씨라고 소개했다.',
}


def build_cases():
    cases = []
    for old in map(json.loads, (ROOT / "data/name_context_v3.jsonl").read_text().splitlines()):
        cases.append(row("v4-development-" + old["id"], "train",
                         "v4-development-" + old["context_id"],
                         old["text"], old["expected"], old["surface"]))
    for split, names in NAMES.items():
        templates = (*POSITIVE[split], MIXED[split])
        for i, template in enumerate(templates):
            for j, name in enumerate(names):
                word = WORDS[split][j % len(WORDS[split])]
                text, expected = render(template.replace("{word}", word), name,
                                        names[(j + 4) % len(names)])
                cases.append(row(f"v4-{split}-person-{i:02}-{j:03}", split,
                                 f"v4-{split}-person-{i:02}", text, expected, name))
        for i, template in enumerate(NEGATIVE[split]):
            for j, word in enumerate(WORDS[split]):
                cases.append(row(f"v4-{split}-negative-{i:02}-{j:02}", split,
                                 f"v4-{split}-negative-{i:02}",
                                 template.format(word=word, **particles(word)), []))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in build_cases())
    if args.check:
        if OUTPUT.read_text(encoding="utf-8") != text:
            raise SystemExit("v4 corpus differs from generator")
    else:
        OUTPUT.write_text(text, encoding="utf-8")
    print(f"{len(build_cases())} v4 synthetic cases")


if __name__ == "__main__":
    main()
