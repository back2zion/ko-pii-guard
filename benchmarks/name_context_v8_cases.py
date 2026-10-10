"""V8: retire inspected v7 data, then freeze new validation and evaluation."""

import argparse
import json
from pathlib import Path

from name_context_v3_cases import particles, render, row

ROOT = Path(__file__).parent
OUTPUT = ROOT / "data/name_context_v8.jsonl"
NAMES = {
    "validation": ("표하린", "도예원", "추해나", "옥지온", "황목시윤", "동방예솔",
                   "림", "뜰", "긍정", "고운", "Vera Example", "Seth Sample"),
    "evaluation": ("박소이", "강나율", "민이안", "신하빈", "도유하", "정예온",
                   "변하윤", "배서진", "남궁루아", "선우이든", "청", "률",
                   "공정", "솔직", "성실함", "담대", "Cora Example", "Neil Sample"),
}
WORDS = {
    "validation": ("긍정", "고운", "단정함", "차분함", "잔잔함", "다정함"),
    "evaluation": ("공정", "솔직", "성실함", "담대", "온화함", "기쁨", "신중함", "따스함"),
}
PERSON = {
    "validation": (
        "수강 접수를 처리한 {a}{a_subject} 신청인 {b} 씨에게 준비물을 안내했다.",
        "증명 자료인 첨부본/{a}_교육이수확인서.pdf를 읽었다.",
        "안내 방송을 맡은 이는 {a}이고 좌석 배치를 도운 이는 {b}이다.",
        "물건을 맡긴 고객은 본인의 성명을 {a}{a_quotative} 밝혔다.",
        '단편집의 제목은 "{word}"이며 지은이는 {a} 씨다.',
        "공예가 {a} 씨와 보조자인 {b} 씨가 재료를 나눴다.",
    ),
    "evaluation": (
        "공연 예약을 도운 {a}{a_subject} 관객 {b} 씨에게 좌석표를 보냈다.",
        "검토 문서에는 위촉본/{a}_위촉확인서.pdf가 들어 있었다.",
        "무대 조율을 맡은 이는 {a}이고 객석 안내를 담당한 이는 {b}이다.",
        "분실물을 문의한 손님은 자신의 성명이 {a}{a_quotative} 설명했다.",
        '소책자의 표제는 "{word}"이며 집필자는 {a} 씨다.',
        "금속공예가 {a} 씨와 제자인 {b} 씨가 작업대를 정돈했다.",
        "주말 접수 담당으로 {a}, {b} 두 사람을 배정했다.",
        "신청인 {a} 씨에게 담당자 {b} 씨가 예약증을 건넸다.",
    ),
}
NEGATIVE = {
    "validation": (
        "교실 게시물의 {word}{topic} 학생 성명 대신 담은 가치 표현이다.",
        "뜻풀이집/{word}_어휘설명.pdf에는 개념 설명을 담았다.",
        "세미나는 추상 명사 {word}의 의미를 정리하는 자리였다.",
        "작품 목록의 {word}{topic} 저자 성명이 아니라 표제어다.",
    ),
    "evaluation": (
        "기관의 표어 {word}{topic} 직원 이름 대신 내건 가치 개념이다.",
        "학습자료 낱말집/{word}_용례풀이.pdf에는 어휘 설명이 있다.",
        "편찬자는 {word}{object} 사람 이름이 아닌 일반 명사로 분류했다.",
        "서평에 적힌 {word}{topic} 필자 성명 대신 책 제목을 가리킨다.",
        "수업 시간에는 추상 개념인 {word}의 쓰임을 비교했다.",
        "안내문에 실린 핵심어 {word}{topic} 특정 인물을 나타내지 않는다.",
    ),
}


def build_cases():
    cases = []
    for old in map(json.loads, (ROOT / "data/name_context_v7.jsonl").read_text().splitlines()):
        cases.append(row("v8-development-" + old["id"], "train",
                         "v8-development-" + old["context_id"],
                         old["text"], old["expected"], old["surface"]))
    for split, names in NAMES.items():
        for i, template in enumerate(PERSON[split]):
            for j, name in enumerate(names):
                text, expected = render(template.replace("{word}", WORDS[split][j % len(
                    WORDS[split]
                )]), name, names[(j + 3) % len(names)])
                cases.append(row(f"v8-{split}-person-{i:02}-{j:03}", split,
                                 f"v8-{split}-person-{i:02}", text, expected, name))
        for i, template in enumerate(NEGATIVE[split]):
            for j, word in enumerate(WORDS[split]):
                cases.append(row(f"v8-{split}-negative-{i:02}-{j:02}", split,
                                 f"v8-{split}-negative-{i:02}",
                                 template.format(word=word, **particles(word)), []))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in build_cases())
    if args.check:
        if OUTPUT.read_text(encoding="utf-8") != text:
            raise SystemExit("v8 data differs from authored generator")
    else:
        OUTPUT.write_text(text, encoding="utf-8")
    print(f"{len(build_cases())} v8 synthetic cases")


if __name__ == "__main__":
    main()
