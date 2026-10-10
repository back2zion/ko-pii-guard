"""V5: retire inspected v4 data, then freeze new validation and evaluation."""

import argparse
import json
from pathlib import Path

from name_context_v3_cases import particles, render, row

ROOT = Path(__file__).parent
OUTPUT = ROOT / "data/name_context_v5.jsonl"
NAMES = {
    "validation": ("황채린", "목유안", "채하진", "복서우", "남궁라온", "선우예빈",
                   "승", "훈", "인내", "배려", "Maya Example", "Leon Sample"),
    "evaluation": ("김소율", "박도하", "이채원", "최서린", "정유겸", "한시온",
                   "장라희", "권이든", "황보예솔", "남궁유빈", "노", "단",
                   "윤택", "온화", "신뢰", "용기", "Nina Sample", "Luca Example"),
}
WORDS = {
    "validation": ("인내", "배려", "화사함", "너그러움", "고즈넉함", "봄기운"),
    "evaluation": ("윤택", "온화", "신뢰", "용기", "생동감", "평정", "상냥함", "친절"),
}
PERSON = {
    "validation": (
        "해설을 맡은 {a}{a_subject} 관람객 {b} 씨를 맞이했다.",
        "이력 확인을 위해 인사/{a}_경력확인서.pdf를 제출했다.",
        "축제를 준비한 이는 {a}이고 안내를 맡은 이는 {b}이다.",
        "우편을 찾으러 온 분은 자신의 이름을 {a}{a_quotative} 밝혔다.",
        '표어 "{word}"를 설명한 발표자는 {a} 씨다.',
        "사서 {a} 씨와 독자인 {b} 씨가 책장을 정리했다.",
    ),
    "evaluation": (
        "견학 안내를 맡은 {a}{a_subject} 신청자 {b} 씨에게 배지를 건넸다.",
        "신청 서류 중 접수/{a}_자격확인서.pdf를 확인했다.",
        "무대를 설계한 이는 {a}이고 조명을 맡은 이는 {b}이다.",
        "입장을 기다리던 손님은 성명이 {a}{a_quotative} 말했다.",
        '기사의 제목은 "{word}"이며 작성자는 {a} 씨다.',
        "서예가 {a} 씨와 수강생인 {b} 씨가 붓을 골랐다.",
        "새로운 진행 요원으로 {a}, {b} 두 사람을 추천했다.",
        "여행객 {a} 씨에게 직원 {b} 씨가 예약서를 전달했다.",
    ),
}
NEGATIVE = {
    "validation": (
        "학급 표어 {word}{topic} 특정인의 이름을 나타내지 않는다.",
        "사전학습/{word}_뜻정리.pdf에는 어휘 해설만 있다.",
        "오늘의 주제 {word}에 대해 개념을 정리했다.",
        "표지의 {word}{topic} 이름이 아니라 에세이 제목이다.",
    ),
    "evaluation": (
        "교실 벽의 문구 {word}{topic} 학생 이름 대신 적은 구호다.",
        "학습용 파일 용어/{word}_어휘풀이.pdf를 수업에서 읽었다.",
        "사전 편찬자는 {word}{object} 보통명사 항목에 넣었다.",
        "추천 도서의 {word}{topic} 필명이나 성함이 아닌 표제이다.",
        "추상적인 뜻을 가진 말 {word}의 쓰임을 비교했다.",
        "학습 안내문의 핵심 개념은 {word}이며 사람을 지칭하지 않는다.",
    ),
}


def build_cases():
    cases = []
    for old in map(json.loads, (ROOT / "data/name_context_v4.jsonl").read_text().splitlines()):
        cases.append(row("v5-development-" + old["id"], "train",
                         "v5-development-" + old["context_id"],
                         old["text"], old["expected"], old["surface"]))
    for split, names in NAMES.items():
        for i, template in enumerate(PERSON[split]):
            for j, name in enumerate(names):
                text, expected = render(template.replace("{word}", WORDS[split][j % len(
                    WORDS[split]
                )]), name, names[(j + 3) % len(names)])
                cases.append(row(f"v5-{split}-person-{i:02}-{j:03}", split,
                                 f"v5-{split}-person-{i:02}", text, expected, name))
        for i, template in enumerate(NEGATIVE[split]):
            for j, word in enumerate(WORDS[split]):
                cases.append(row(f"v5-{split}-negative-{i:02}-{j:02}", split,
                                 f"v5-{split}-negative-{i:02}",
                                 template.format(word=word, **particles(word)), []))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in build_cases())
    if args.check:
        if OUTPUT.read_text(encoding="utf-8") != text:
            raise SystemExit("v5 data differs from authored generator")
    else:
        OUTPUT.write_text(text, encoding="utf-8")
    print(f"{len(build_cases())} v5 synthetic cases")


if __name__ == "__main__":
    main()
