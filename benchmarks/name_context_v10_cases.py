"""V10: retire inspected v10 data, then freeze new validation and evaluation."""

import argparse
import json
from pathlib import Path

from name_context_v3_cases import particles, render, row

ROOT = Path(__file__).parent
OUTPUT = ROOT / "data/name_context_v10.jsonl"
NAMES = {
    "validation": ("금소휘", "뇌여울", "해온빈", "동이루", "남궁서봄", "선우예결",
                   "쾌", "슘", "화목", "응원", "Nia Trial", "Gus Trial"),
    "evaluation": ("기윤솔", "설온겸", "팽이든", "봉하윤", "추리안", "온서경",
                   "갈유담", "사채온", "독고윤슬", "사공서결", "쨈", "퀸",
                   "공감", "기개", "도전", "차분함", "Ivo Specimen", "Rhea Specimen"),
}
WORDS = {
    "validation": ("화목", "응원", "아늑함", "담담함", "청명함", "투명함"),
    "evaluation": ("공감", "기개", "도전", "차분함", "유쾌함", "산뜻함", "단아함", "넉넉함"),
}
PERSON = {
    "validation": (
        "식순 점검을 도운 {a}{a_subject} 참석자 {b} 씨에게 순서표를 건넸다.",
        "보관 상자에서 점검분/{a}_참가내역.pdf를 꺼냈다.",
        "창구 인수인은 {a}이고 운행 기록 작성인은 {b}이다.",
        "상담을 신청한 분이 성함을 {a}{a_quotative} 말씀하셨다.",
        '화집에 쓰인 문구는 "{word}"이며 삽화가는 {a} 씨다.',
        "공예인 {a} 씨와 견습생 {b} 씨가 찰흙을 빚었다.",
    ),
    "evaluation": (
        "접수 현황을 확인한 {a}{a_subject} 내방객 {b} 씨에게 번호표를 주었다.",
        "수합한 목록에서 보존분/{a}_위임기록.pdf를 선택했다.",
        "서류를 인수할 이는 {a}이고 수량을 검수할 이는 {b}이다.",
        "신청을 마친 방문자는 자신의 성함을 {a}{a_quotative} 밝혔다.",
        '노트에 적힌 구절은 "{word}"이고 필기한 사람은 {a} 씨다.',
        "도예가 {a} 씨와 수강생 {b} 씨가 가마를 살폈다.",
        "저녁 상담 창구 근무자로 {a}, {b} 두 명을 배정했다.",
        "수령인 {a} 씨에게 전달 담당자 {b} 씨가 상자를 건넸다.",
    ),
}
NEGATIVE = {
    "validation": (
        "포스터 속 {word}{topic} 주민 성명이 아닌 생활 덕목이다.",
        "사전초안/{word}_개념정리.pdf는 용어 설명 자료다.",
        "강의에서 다룬 추상 명사 {word}{topic} 인명과 무관하다.",
        "도서 목록의 {word}{topic} 저자 성명이 아니라 책의 표제다.",
    ),
    "evaluation": (
        "벽보에 인쇄한 {word}{topic} 참가자 성명이 아닌 실천 덕목이다.",
        "용어 수업의 뜻모음/{word}_정의해설.pdf를 펼쳤다.",
        "편집자는 {word}{object} 사람을 가리키지 않는 추상 명사라고 풀이했다.",
        "장서 대장의 {word}{topic} 필자 이름 대신 책 제목을 나타낸다.",
        "토론 자료에서는 {word}라는 개념이 가진 뜻을 비교했다.",
        "학습지의 낱말 {word}{topic} 특정인의 성명으로 쓰이지 않았다.",
    ),
}

def build_cases():
    cases = []
    for old in map(json.loads, (ROOT / "data/name_context_v9.jsonl").read_text().splitlines()):
        cases.append(row("v10-development-" + old["id"], "train",
                         "v10-development-" + old["context_id"],
                         old["text"], old["expected"], old["surface"]))
    for split, names in NAMES.items():
        for i, template in enumerate(PERSON[split]):
            for j, name in enumerate(names):
                text, expected = render(template.replace("{word}", WORDS[split][j % len(
                    WORDS[split]
                )]), name, names[(j + 3) % len(names)])
                cases.append(row(f"v10-{split}-person-{i:02}-{j:03}", split,
                                 f"v10-{split}-person-{i:02}", text, expected, name))
        for i, template in enumerate(NEGATIVE[split]):
            for j, word in enumerate(WORDS[split]):
                cases.append(row(f"v10-{split}-negative-{i:02}-{j:02}", split,
                                 f"v10-{split}-negative-{i:02}",
                                 template.format(word=word, **particles(word)), []))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in build_cases())
    if args.check:
        if OUTPUT.read_text(encoding="utf-8") != text:
            raise SystemExit("v10 data differs from authored generator")
    else:
        OUTPUT.write_text(text, encoding="utf-8")
    print(f"{len(build_cases())} v10 synthetic cases")


if __name__ == "__main__":
    main()
