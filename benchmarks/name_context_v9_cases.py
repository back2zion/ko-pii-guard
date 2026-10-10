"""V9: retire inspected v8 data, then freeze new validation and evaluation."""

import argparse
import json
from pathlib import Path

from name_context_v3_cases import particles, render, row

ROOT = Path(__file__).parent
OUTPUT = ROOT / "data/name_context_v9.jsonl"
NAMES = {
    "validation": ("호서린", "우예담", "여지온", "주하린", "남궁이솔", "선우채하",
                   "늠", "묵", "기대", "용서", "Ada Demo", "Oren Demo"),
    "evaluation": ("허유리", "김채민", "손해온", "문예솔", "탁서준", "채하은",
                   "편다빈", "구서온", "제갈유담", "황보채림", "훤", "범",
                   "믿음", "감사", "정진", "신중함", "Ari Sample", "Jade Example"),
}
WORDS = {
    "validation": ("기대", "용서", "평온함", "부드러움", "명료함", "포근함"),
    "evaluation": ("믿음", "감사", "정진", "신중함", "경쾌함", "정갈함", "진중함", "유연함"),
}
PERSON = {
    "validation": (
        "예약 확인을 도운 {a}{a_subject} 방문객 {b} 씨에게 표를 전달했다.",
        "정리 중인 경로 확인본/{a}_자격확인서.pdf를 열어 봤다.",
        "진행 멘트를 맡은 이는 {a}이고 현장 정리를 도운 이는 {b}이다.",
        "창구에 온 손님은 자기 성명을 {a}{a_quotative} 알렸다.",
        '전시물의 표제는 "{word}"이며 만든 이는 {a} 씨다.',
        "세공사 {a} 씨와 조력자인 {b} 씨가 도구를 점검했다.",
    ),
    "evaluation": (
        "견본 발송을 맡은 {a}{a_subject} 수신인 {b} 씨에게 운송장을 전달했다.",
        "제출 목록에서 인계본/{a}_수습확인서.pdf를 찾아 읽었다.",
        "기기 설정을 맡은 이는 {a}이고 음량 조절을 담당한 이는 {b}이다.",
        "접수를 기다리는 고객은 본인 이름이 {a}{a_quotative} 답변했다.",
        '수첩의 표제는 "{word}"이며 기록자는 {a} 씨다.',
        "판화가 {a} 씨와 문하생인 {b} 씨가 종이를 골랐다.",
        "오후 민원 안내 담당으로 {a}, {b} 두 사람을 지정했다.",
        "방청인 {a} 씨에게 안내자 {b} 씨가 좌석 위치를 알려줬다.",
    ),
}
NEGATIVE = {
    "validation": (
        "기념품의 문구 {word}{topic} 수령자의 이름 대신 적은 표어다.",
        "개념집/{word}_용어해설.pdf에는 일반 어휘의 뜻이 있다.",
        "독서회에서는 추상 어휘인 {word}의 의미를 논했다.",
        "추천 목록의 {word}{topic} 지은이 이름이 아니라 작품 제목이다.",
    ),
    "evaluation": (
        "학교 현수막의 {word}{topic} 교직원 이름이 아닌 핵심 가치어다.",
        "어휘학습 낱말모음/{word}_의미풀이.pdf를 살펴보았다.",
        "사전 편찬진은 {word}{object} 인명이 아닌 보통명사로 설명했다.",
        "소장 목록에 있는 {word}{topic} 작가 이름이 아니라 표제어다.",
        "발제문은 {word}라는 추상적 개념의 뜻을 다뤘다.",
        "자료의 중심 낱말 {word}{topic} 어떤 사람의 이름도 지칭하지 않는다.",
    ),
}


def build_cases():
    cases = []
    for old in map(json.loads, (ROOT / "data/name_context_v8.jsonl").read_text().splitlines()):
        cases.append(row("v9-development-" + old["id"], "train",
                         "v9-development-" + old["context_id"],
                         old["text"], old["expected"], old["surface"]))
    for split, names in NAMES.items():
        for i, template in enumerate(PERSON[split]):
            for j, name in enumerate(names):
                text, expected = render(template.replace("{word}", WORDS[split][j % len(
                    WORDS[split]
                )]), name, names[(j + 3) % len(names)])
                cases.append(row(f"v9-{split}-person-{i:02}-{j:03}", split,
                                 f"v9-{split}-person-{i:02}", text, expected, name))
        for i, template in enumerate(NEGATIVE[split]):
            for j, word in enumerate(WORDS[split]):
                cases.append(row(f"v9-{split}-negative-{i:02}-{j:02}", split,
                                 f"v9-{split}-negative-{i:02}",
                                 template.format(word=word, **particles(word)), []))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in build_cases())
    if args.check:
        if OUTPUT.read_text(encoding="utf-8") != text:
            raise SystemExit("v9 data differs from authored generator")
    else:
        OUTPUT.write_text(text, encoding="utf-8")
    print(f"{len(build_cases())} v9 synthetic cases")


if __name__ == "__main__":
    main()
