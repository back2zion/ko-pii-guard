"""V7: retire inspected v6 data, then freeze new validation and evaluation."""

import argparse
import json
from pathlib import Path

from name_context_v3_cases import particles, render, row

ROOT = Path(__file__).parent
OUTPUT = ROOT / "data/name_context_v7.jsonl"
NAMES = {
    "validation": ("장예솔", "채라온", "유해준", "민서온", "남궁예하", "선우라희",
                   "핀", "준", "찬란", "여유", "Tara Example", "Noel Sample"),
    "evaluation": ("서다인", "임채온", "안예담", "강도하", "오지안", "진서율",
                   "백시우", "한루아", "황보예림", "제갈하진", "설", "루",
                   "약속", "신념", "존중", "진심", "Evan Sample", "Lena Example"),
}
WORDS = {
    "validation": ("찬란", "여유", "화목", "평안", "햇살", "배려"),
    "evaluation": ("약속", "신념", "존중", "진심", "온기", "신뢰", "공감", "우정"),
}
PERSON = {
    "validation": (
        "견본을 전달한 {a}{a_subject} 담당자 {b} 씨에게 접수 방법을 설명했다.",
        "증빙을 검토하며 제출물/{a}_이력확인서.pdf를 열었다.",
        "소품 제작을 맡은 이는 {a}이고 현장 진행을 도운 이는 {b}이다.",
        "대기 중인 관객은 자기 이름을 {a}{a_quotative} 소개했다.",
        '논집의 표제는 "{word}"이며 편저자는 {a} 씨다.',
        "사진가 {a} 씨와 조수인 {b} 씨가 렌즈를 챙겼다.",
    ),
    "evaluation": (
        "도서 대출을 도운 {a}{a_subject} 이용자 {b} 씨에게 반납일을 알렸다.",
        "제출 문서 가운데 접수본/{a}_연수확인서.pdf를 살펴봤다.",
        "음향 점검을 맡은 이는 {a}이고 영상 재생을 담당한 이는 {b}이다.",
        "보관물을 찾는 손님은 본인의 이름이 {a}{a_quotative} 대답했다.",
        '기획전의 이름은 "{word}"이며 기획자는 {a} 씨다.',
        "목공가 {a} 씨와 견습생인 {b} 씨가 나무를 골랐다.",
        "야간 안내 담당으로 {a}, {b} 두 사람을 선정했다.",
        "참관인 {a} 씨에게 인솔자 {b} 씨가 출입증을 건넸다.",
    ),
}
NEGATIVE = {
    "validation": (
        "행사의 슬로건 {word}{topic} 참석자의 성명과 관련 없는 문구다.",
        "학습노트/{word}_개념풀이.pdf에는 사전 설명이 실려 있다.",
        "강의에서 {word}의 추상적인 뜻을 함께 논의했다.",
        "진열 목록의 {word}{topic} 작가가 아니라 책의 표제다.",
    ),
    "evaluation": (
        "인쇄물의 표어 {word}{topic} 관계자 이름 대신 넣은 가치어다.",
        "학습 파일 어휘집/{word}_용례모음.pdf를 펼쳐 보았다.",
        "사전 집필자는 {word}{object} 일반 명사로 설명했다.",
        "전시 안내에 적힌 {word}{topic} 참여자 성함이 아니라 주제다.",
        "이번 독서 모임에서는 추상적인 개념 {word}의 의미를 살폈다.",
        "포스터에 강조한 문구는 {word}이며 어떤 사람도 가리키지 않는다.",
    ),
}


def build_cases():
    cases = []
    for old in map(json.loads, (ROOT / "data/name_context_v6.jsonl").read_text().splitlines()):
        cases.append(row("v7-development-" + old["id"], "train",
                         "v7-development-" + old["context_id"],
                         old["text"], old["expected"], old["surface"]))
    for split, names in NAMES.items():
        for i, template in enumerate(PERSON[split]):
            for j, name in enumerate(names):
                text, expected = render(template.replace("{word}", WORDS[split][j % len(
                    WORDS[split]
                )]), name, names[(j + 3) % len(names)])
                cases.append(row(f"v7-{split}-person-{i:02}-{j:03}", split,
                                 f"v7-{split}-person-{i:02}", text, expected, name))
        for i, template in enumerate(NEGATIVE[split]):
            for j, word in enumerate(WORDS[split]):
                cases.append(row(f"v7-{split}-negative-{i:02}-{j:02}", split,
                                 f"v7-{split}-negative-{i:02}",
                                 template.format(word=word, **particles(word)), []))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in build_cases())
    if args.check:
        if OUTPUT.read_text(encoding="utf-8") != text:
            raise SystemExit("v7 data differs from authored generator")
    else:
        OUTPUT.write_text(text, encoding="utf-8")
    print(f"{len(build_cases())} v7 synthetic cases")


if __name__ == "__main__":
    main()
