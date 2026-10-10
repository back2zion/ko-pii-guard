"""Author-controlled v2 data: previous inspected data is development only.

New validation/evaluation names, literal sentences and context IDs are disjoint
from each other and all training targets. Filename, multiple-person and one-char
name cases are explicit training categories, not runtime spelling rules.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from name_context_training_cases import NAMES as OLD_NAMES
from name_context_training_cases import particles

ROOT = Path(__file__).parent
OUTPUT = ROOT / "data" / "name_context_v2.jsonl"
TRAIN_NAMES = tuple(
    dict.fromkeys(
        [name for names in OLD_NAMES.values() for name in names]
        + [
            "김철수",
            "김민수",
            "박영희",
            "이지안",
            "최유진",
            "서예준",
            "천다은",
            "김지은",
            "은",
            "이",
            "나",
            "해",
            "솔",
            "강",
            "산",
            "달",
            "봄",
            "빛",
            "결",
            "빈",
            "담",
            "율",
            "Anna Sample",
            "Mina Example",
            "Alex Example",
            "Jose\u0301 Example",
            "Ana·Example",
        ]
    )
)
NAMES = {
    "train": TRAIN_NAMES,
    "validation": (
        "정라율",
        "위서온",
        "유지완",
        "석해나",
        "독고루빈",
        "어금재온",
        "온",
        "린",
        "다",
        "진",
        "여운",
        "행운",
        "John Demo",
        "Anna Mock",
    ),
    "evaluation": (
        "방시우",
        "표예림",
        "진로하",
        "함지유",
        "황목유진",
        "서문하린",
        "소봉이안",
        "동방소율",
        "찬",
        "엘",
        "수",
        "윤",
        "열매",
        "도움",
        "은혜",
        "성실",
        "Sara Dummy",
        "Ravi Mock",
        "한나·예온",
        "Noe\u0308l Demo",
    ),
}
TEMPLATES = {
    "train": (
        "직원 {a}{a_topic} 방금 인사를 했다.",
        "선생님은 {a} 씨를 소개했다.",
        "성함을 묻자 {a}{a_quotative} 대답했다.",
        "{a} 씨와 {b} 씨가 서류를 함께 검토했습니다.",
        "고객 {a}{a_subject} 담당자 {b} 씨에게 연락했다.",
        "회의록은 {a} 씨가 작성하고 {b} 씨가 확인했다.",
        "접수 명단에는 {a}, {b} 두 사람의 이름이 적혀 있다.",
        "{a}{a_topic} {b} 씨의 답변을 기다렸다.",
        "동료 {a}{a_genitive} 제안을 {b} 씨가 검토했다.",
        "첨부: {a}_이력서.pdf",
        "지원자 {a} 씨의 파일은 {a}_신청서.pdf 입니다.",
        "제출된 문서: {a}_경력기술서.docx",
        "이력서/{a}_자기소개서.txt",
        "오늘 받은 {a}_동의서.pdf를 확인했다.",
        "저장된 지원서의 파일명은 {a}.pdf이다.",
        "명단: {a}, {b}",
        "손님 {a} 씨가 도착했습니다. {b} 씨에게 알려 주세요.",
        "신청자인 {a}{a_subject} 확인 서명을 했다.",
        "상담을 신청한 분은 {a}이고 동행인은 {b}이다.",
        "방문한 사람은 {a}{a_quotative} 자신을 소개했다.",
        "오늘 만난 동료 {a}{a_genitive} 연락을 기다렸다.",
        "{a} 씨에게 {b}_이력서.pdf를 전달했다.",
    ),
    "validation": (
        "조리사 {a} 씨에게 다음 식단을 물었다.",
        "검수를 마친 {a}{a_topic} 승인 여부를 통보했다.",
        "회원 가입을 도와준 사람의 이름은 {a}이었다.",
        "구매자 {a}{a_subject} 판매자인 {b} 씨에게 문의를 남겼다.",
        "수업을 진행한 {a} 씨와 보조한 {b} 씨가 함께 퇴실했다.",
        "자료를 작성한 사람은 {a}이며 검토한 사람은 {b}이다.",
        "검토할 첨부 문서의 경로는 신청/{a}_접수증.pdf 이다.",
        "경력 확인용 {a}_포트폴리오.pdf를 열었다.",
        "등록자: {a} / 추천자: {b}",
        "진행자가 {a} 씨에게 {b} 씨의 의견을 전달했다.",
    ),
    "evaluation": (
        "기차에 탑승한 승객 {a} 씨는 승무원을 불렀다.",
        "택배를 찾으러 온 {a}{a_subject} 수령 확인란에 서명했다.",
        "이 물건을 맡긴 분은 본인 이름이 {a}{a_quotative} 말했다.",
        "사진작가 {a}{a_topic} 모델 {b} 씨와 촬영 시간을 정했다.",
        "사서 {a} 씨와 독자인 {b} 씨는 대출 기한을 논의했다.",
        "낭독을 맡은 사람은 {a}이고 반주자는 {b}이다.",
        "내려받은 문서 보관/{a}_재직증명서.pdf를 열람했다.",
        "메일에는 {a}_교육수료증.pdf가 첨부되어 있었다.",
        "청소 당번으로 {a}, {b} 두 사람이 정해졌다.",
        "길 안내를 부탁한 {a} 씨에게 {b} 씨가 약도를 그려 주었다.",
    ),
}
# Explicitly non-person uses are separate authored templates. File titles with
# common nouns are negative only when their non-person meaning is unambiguous.
WORDS = {
    "train": (
        "하나",
        "사랑",
        "미소",
        "이상",
        "소망",
        "보람",
        "희망",
        "봄",
        "빛",
        "바다",
        "하늘",
        "나무",
        "가을",
        "여름",
        "겨울",
        "별",
        "지혜",
        "온기",
    ),
    "validation": ("행운", "여운", "설렘", "햇살", "바람", "행복"),
    "evaluation": ("도움", "은혜", "성실", "열매", "찬란함", "온화함"),
}
NEGATIVE_TEMPLATES = {
    "train": (
        "이 글은 {word}의 의미를 설명한다.",
        "작품의 주제는 {word}이다.",
        "단어 {word}{topic} 명사가 쓰인 예문이다.",
        '문서 제목은 "{word}"이며 작성자의 이름은 기록하지 않았다.',
        '표지에 적힌 "{word}"는 책의 제목이다.',
        "이번 강연은 {word}에 관한 이야기였다.",
        "글에서 {word}{object} 강조했지만 사람을 지칭하지는 않는다.",
        "자료를 {word}라는 이름의 폴더에 보관했다.",
    ),
    "validation": (
        "{word}라는 개념에 대해 토론했다.",
        '전시품의 이름은 "{word}"이며 특정 인물을 나타내지 않는다.',
        "학생들은 {word}에 관한 에세이를 읽었다.",
        "이 노래의 제목에 {word}{subject} 들어간다.",
    ),
    "evaluation": (
        "사전에서 {word}라는 낱말의 정의를 찾아보았다.",
        "이번 수업의 학습 주제는 {word}에 관한 설명이다.",
        "작가는 수필에서 {word}{object} 추상적인 개념으로 다뤘다.",
        "발표 자료에 적힌 {word}{topic} 사람 이름이 아닌 표제어였다.",
    ),
}
EXTRA_NEGATIVES = {
    "train": (
        "두 의견을 하나로 합쳐 정리했다.",
        "우선 질문 하나를 적어 보자.",
        "예산 항목 하나를 삭제했다.",
        "중요한 준비물 하나가 빠져 있었다.",
        "남은 질문 하나를 정리했다.",
        "아이는 미소를 지으며 손을 흔들었다.",
        "새해의 소망은 가족 모두의 건강이다.",
        "조각상의 미소는 보는 방향에 따라 달라 보였다.",
        "첨부: 채용_안내.pdf",
        "첨부: 개인정보_처리방침.pdf",
        "첨부: 이용_약관.pdf",
        "첨부: 회의_자료.pdf",
        "고객님, 검토가 완료되었습니다.",
        "감사합니다. 좋은 하루 보내세요.",
        "성명 입력 필요",
        "성명: 미기재",
        "담당자: 미정",
    ),
    "validation": (
        "첨부: 서비스_소개.pdf",
        "첨부: 운영_지침.pdf",
        "모든 자료를 읽고 나서 답변해 주세요.",
        "새로운 일정이 확정되었습니다.",
    ),
    "evaluation": (
        "첨부: 안전_수칙.pdf",
        "첨부: 참가_안내.pdf",
        "가장 적절한 선택지 하나를 골랐다.",
        "밝은 미소로 감사의 뜻을 전했다.",
        "정기 점검에서 특별한 이상은 발견되지 않았다.",
        "올해 이루고 싶은 소망을 달력에 적었다.",
        "요청하신 정보는 아직 준비되지 않았습니다.",
        "앞으로의 일정은 추후 공지합니다.",
    ),
}


def render(template, a, b):
    """Offsets come from placeholders, including repeated names and two people."""
    values = {"a": a, "b": b}
    values.update({f"a_{k}": v for k, v in particles(a).items()})
    values.update({f"b_{k}": v for k, v in particles(b).items()})
    parts, expected, position = [], [], 0
    for part in re.split(r"(\{[^}]+\})", template):
        key = part[1:-1] if part.startswith("{") else None
        value = values[key] if key else part
        if key in ("a", "b"):
            expected.append(dict(entity="KR_NAME", start=position, end=position + len(value)))
        parts.append(value)
        position += len(value)
    return "".join(parts), expected


def build_cases():
    rows = []
    for split, names in NAMES.items():
        for i, template in enumerate(TEMPLATES[split]):
            for j, name in enumerate(names):
                other = names[(j + 3) % len(names)]
                text, expected = render(template, name, other)
                rows.append(
                    dict(
                        id=f"v2-{split}-person-{i:02}-{j:03}",
                        split=split,
                        context_id=f"v2-{split}-person-{i:02}",
                        surfaces=sorted({text[e["start"] : e["end"]] for e in expected}),
                        surface=name,
                        track="prose",
                        text=text,
                        expected=expected,
                    )
                )
        for i, template in enumerate(NEGATIVE_TEMPLATES[split]):
            for j, word in enumerate(WORDS[split]):
                text = template.format(word=word, **particles(word))
                rows.append(
                    dict(
                        id=f"v2-{split}-negative-{i:02}-{j:02}",
                        split=split,
                        context_id=f"v2-{split}-negative-{i:02}",
                        surfaces=[],
                        surface=None,
                        track="prose",
                        text=text,
                        expected=[],
                    )
                )
        for i, text in enumerate(EXTRA_NEGATIVES[split]):
            rows.append(
                dict(
                    id=f"v2-{split}-extra-negative-{i:02}",
                    split=split,
                    context_id=f"v2-{split}-extra-negative-{i:02}",
                    surfaces=[],
                    surface=None,
                    track="prose",
                    text=text,
                    expected=[],
                )
            )
    # Every inspected v1 split is now DEVELOPMENT training data. Preserve IDs
    # via a prefix and preserve the original text and gold labels.
    for file in (
        "name_context_training.jsonl",
        "name_address.jsonl",
        "business_korean.jsonl",
        "name_field_boundaries.jsonl",
    ):
        for old in map(json.loads, (ROOT / "data" / file).read_text().splitlines()):
            expected = [e for e in old["expected"] if e["entity"] == "KR_NAME"]
            rows.append(
                dict(
                    id=f"v2-development-{file}-{old['id']}",
                    split="train",
                    context_id=f"v2-development-{file}-{old.get('context_id', old['id'])}",
                    surfaces=sorted({old["text"][e["start"] : e["end"]] for e in expected}),
                    surface=old.get("surface"),
                    track="prose",
                    text=old["text"],
                    expected=expected,
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in build_cases())
    if args.check:
        if OUTPUT.read_text(encoding="utf-8") != text:
            raise SystemExit("v2 data differs from authored generator")
    else:
        OUTPUT.write_text(text, encoding="utf-8")
    print(f"{len(build_cases())} synthetic v2 cases")


if __name__ == "__main__":
    main()
