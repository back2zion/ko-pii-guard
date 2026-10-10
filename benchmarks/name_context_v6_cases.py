"""V6: retire inspected v5 data, then freeze new validation and evaluation."""

import argparse
import json
from pathlib import Path

from name_context_v3_cases import particles, render, row

ROOT = Path(__file__).parent
OUTPUT = ROOT / "data/name_context_v6.jsonl"
NAMES = {
    "validation": ("손예나", "우하온", "천서윤", "노예솔", "제갈서우", "독고예진",
                   "겸", "휘", "낙관", "성원", "Rosa Example", "Liam Sample"),
    "evaluation": ("문예린", "서하준", "심채율", "안도윤", "오서아", "류지후",
                   "정선우", "곽해린", "황목예담", "동방하율", "람", "해솔",
                   "의리", "정의", "성의", "헌신", "Mira Sample", "Hugo Example"),
}
WORDS = {
    "validation": ("낙관", "성원", "화사함", "너그러움", "고즈넉함", "봄기운"),
    "evaluation": ("의리", "정의", "성의", "헌신", "생동감", "평정", "상냥함", "친절"),
}
PERSON = {
    "validation": (
        "해설을 진행한 {a}{a_subject} 방문객 {b} 씨를 맞이했다.",
        "등록 확인을 위해 기록/{a}_경력확인서.pdf를 제출했다.",
        "축제를 기획한 이는 {a}이고 접수를 도운 이는 {b}이다.",
        "신분증을 찾으러 온 분은 자신의 이름을 {a}{a_quotative} 밝혔다.",
        '표어 "{word}"를 설명한 발표자는 {a} 씨다.',
        "점원 {a} 씨와 구매자인 {b} 씨가 구매 내역을 확인했다.",
    ),
    "evaluation": (
        "견학 접수를 도운 {a}{a_subject} 신청자 {b} 씨에게 참가증을 전달했다.",
        "신청 서류 중 지원/{a}_참여확인서.pdf를 확인했다.",
        "영상 편집을 맡은 이는 {a}이고 촬영을 담당한 이는 {b}이다.",
        "순서를 기다리던 손님은 성명이 {a}{a_quotative} 말했다.",
        '수필의 제목은 "{word}"이며 작성자는 {a} 씨다.',
        "조각가 {a} 씨와 수강생인 {b} 씨가 도구를 정리했다.",
        "새로운 봉사 요원으로 {a}, {b} 두 사람을 추천했다.",
        "방문자 {a} 씨에게 직원 {b} 씨가 안내서를 건넸다.",
    ),
}
NEGATIVE = {
    "validation": (
        "학교의 표어 {word}{topic} 특정인의 이름을 나타내지 않는다.",
        "용어자료/{word}_뜻풀이.pdf에는 어휘 해설만 있다.",
        "수업의 주제어 {word}에 대해 개념을 정리했다.",
        "목록의 {word}{topic} 이름이 아니라 에세이 제목이다.",
    ),
    "evaluation": (
        "복도 벽의 문구 {word}{topic} 학생 이름 대신 적은 구호다.",
        "교재 파일 낱말/{word}_의미해설.pdf를 수업에서 읽었다.",
        "교재 편집자는 {word}{object} 보통명사 항목에 넣었다.",
        "서가 목록의 {word}{topic} 필명이나 성함이 아닌 표제이다.",
        "추상 의미를 나타내는 낱말 {word}의 쓰임을 비교했다.",
        "토론 안내문의 핵심 주제은 {word}이며 사람을 지칭하지 않는다.",
    ),
}


def build_cases():
    cases = []
    for old in map(json.loads, (ROOT / "data/name_context_v5.jsonl").read_text().splitlines()):
        cases.append(row("v6-development-" + old["id"], "train",
                         "v6-development-" + old["context_id"],
                         old["text"], old["expected"], old["surface"]))
    for split, names in NAMES.items():
        for i, template in enumerate(PERSON[split]):
            for j, name in enumerate(names):
                text, expected = render(template.replace("{word}", WORDS[split][j % len(
                    WORDS[split]
                )]), name, names[(j + 3) % len(names)])
                cases.append(row(f"v6-{split}-person-{i:02}-{j:03}", split,
                                 f"v6-{split}-person-{i:02}", text, expected, name))
        for i, template in enumerate(NEGATIVE[split]):
            for j, word in enumerate(WORDS[split]):
                cases.append(row(f"v6-{split}-negative-{i:02}-{j:02}", split,
                                 f"v6-{split}-negative-{i:02}",
                                 template.format(word=word, **particles(word)), []))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in build_cases())
    if args.check:
        if OUTPUT.read_text(encoding="utf-8") != text:
            raise SystemExit("v6 data differs from authored generator")
    else:
        OUTPUT.write_text(text, encoding="utf-8")
    print(f"{len(build_cases())} v6 synthetic cases")


if __name__ == "__main__":
    main()
