"""V10: retire inspected v11 data, then freeze new validation and evaluation."""

import argparse
import json
from pathlib import Path

from name_context_v3_cases import particles, render, row

ROOT = Path(__file__).parent
OUTPUT = ROOT / "data/name_context_v11.jsonl"
NAMES = {
    "validation": ("견소은", "복해솔", "석여린", "양온율", "남궁채루", "선우가람",
                   "쨍", "쫑", "화합", "포용", "Milo Trial", "Lena Trial"),
    "evaluation": ("방이겸", "형서이", "도유결", "풍나윤", "연혜빈", "태리솔",
                   "성채울", "육하결", "독고해온", "사공온해", "쏨", "깃",
                   "열정", "성찰", "설렘", "소박함", "Zeno Sample", "Etta Sample"),
}
WORDS = {
    "validation": ("화합", "포용", "순박함", "수수함", "다정함", "정감"),
    "evaluation": ("열정", "성찰", "설렘", "소박함", "진정함", "솔직함", "참됨", "협력"),
}
PERSON = {
    "validation": (
        "물품 배분 담당 {a}{a_subject} 요청자 {b} 씨에게 수량을 알렸다.",
        "기록함에서 전달분/{a}_연수확인서.pdf를 확인했다.",
        "반납 담당자는 {a}이고 접수 담당자는 {b}이다.",
        "방문 신청인은 자기 이름을 {a}{a_quotative} 소개했다.",
        '카드의 문장은 "{word}"이며 쓴 사람은 {a} 씨다.',
        "조각가 {a} 씨와 조수 {b} 씨가 작업대를 정리했다.",
    ),
    "evaluation": (
        "현장 접수를 맡은 {a}{a_subject} 신청자 {b} 씨에게 접수증을 건넸다.",
        "담당자는 열람본/{a}_경력확인서.pdf를 첨부 목록에서 골랐다.",
        "봉투를 전달할 사람은 {a}이고 서명을 받을 사람은 {b}이다.",
        "상담실을 찾은 고객은 성명이 {a}{a_quotative} 말했다.",
        '책갈피의 문구는 "{word}"이며 작성한 이는 {a} 씨다.',
        "사진가 {a} 씨와 동료 {b} 씨가 조명을 옮겼다.",
        "당일 안내를 맡을 직원으로 {a}, {b} 두 사람을 정했다.",
        "방문객 {a} 씨가 진행 요원 {b} 씨에게 표를 보여줬다.",
    ),
}
NEGATIVE = {
    "validation": (
        "안내문의 {word}{topic} 직원 이름이 아닌 공동체 가치다.",
        "개념모음/{word}_낱말풀이.pdf는 어휘 교육 자료다.",
        "발표에서 {word}{object} 인명 대신 추상 어휘로 다뤘다.",
        "서가 표지의 {word}{topic} 저자명이 아닌 도서 제목이다.",
    ),
    "evaluation": (
        "행사 표어의 {word}{topic} 누군가의 이름 대신 지향 가치를 뜻한다.",
        "어휘 설명 자료인 뜻풀이/{word}_용어정리.pdf를 열었다.",
        "교재는 {word}{object} 인물이 아닌 보통명사의 사례로 제시했다.",
        "전시 목록의 {word}{topic} 출품자 성명이 아니라 작품의 표제다.",
        "논의의 주제는 추상 명사 {word}가 나타내는 의미였다.",
        "문장에 쓰인 {word}{topic} 사람의 이름으로 해석하지 않는 일반 단어다.",
    ),
}

def build_cases():
    cases = []
    for old in map(json.loads, (ROOT / "data/name_context_v10.jsonl").read_text().splitlines()):
        cases.append(row("v11-development-" + old["id"], "train",
                         "v11-development-" + old["context_id"],
                         old["text"], old["expected"], old["surface"]))
    for split, names in NAMES.items():
        for i, template in enumerate(PERSON[split]):
            for j, name in enumerate(names):
                text, expected = render(template.replace("{word}", WORDS[split][j % len(
                    WORDS[split]
                )]), name, names[(j + 3) % len(names)])
                cases.append(row(f"v11-{split}-person-{i:02}-{j:03}", split,
                                 f"v11-{split}-person-{i:02}", text, expected, name))
        for i, template in enumerate(NEGATIVE[split]):
            for j, word in enumerate(WORDS[split]):
                cases.append(row(f"v11-{split}-negative-{i:02}-{j:02}", split,
                                 f"v11-{split}-negative-{i:02}",
                                 template.format(word=word, **particles(word)), []))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in build_cases())
    if args.check:
        if OUTPUT.read_text(encoding="utf-8") != text:
            raise SystemExit("v11 data differs from authored generator")
    else:
        OUTPUT.write_text(text, encoding="utf-8")
    print(f"{len(build_cases())} v11 synthetic cases")


if __name__ == "__main__":
    main()
