"""Author-controlled synthetic name training and disjoint evaluation cases.

No model calls or predictions. Existing name-context cases are now development
training data, never held-out evidence. New splits separate target names and
person templates; synthetic crossed examples are not independent field samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent
OUTPUT = ROOT / "data" / "name_context_training.jsonl"

NAMES = {
    "train": (
        "강가온",
        "문다솜",
        "한서후",
        "윤아린",
        "송지안",
        "백해준",
        "구나래",
        "허유찬",
        "심라희",
        "차이든",
        "유다인",
        "임서안",
        "남궁서후",
        "황보가온",
        "제갈유찬",
        "선우다솜",
        "독고지안",
        "사공라희",
        "탁온유",
        "견시온",
        "뇌아린",
        "빙예담",
        "즙하율",
        "어금해준",
        "하나",
        "사랑",
        "미소",
        "이상",
        "소망",
        "보람",
        "희망",
        "새봄",
        "가을",
        "한별",
        "여름",
        "겨울",
        "바다",
        "하늘",
        "나무",
        "빛",
        "봄",
        "Mina Example",
        "Anna Sample",
        "Nguyễn Test",
        "Jose\u0301 Sample",
        "안나·가온",
    ),
    "validation": (
        "양다예",
        "배현오",
        "옥지율",
        "도은재",
        "남궁라임",
        "황보이안",
        "기쁨",
        "평화",
        "노을",
        "한결",
        "Evan Demo",
        "류하은",
    ),
    "evaluation": (
        "전유담",
        "장하준",
        "길예온",
        "마은서",
        "독고로아",
        "제갈하늬",
        "어금은솔",
        "선우초연",
        "설레임",
        "슬기",
        "누리",
        "온기",
        "지혜",
        "별",
        "나윤·서연",
        "Lina Mock",
    ),
}
TEMPLATES = {
    "train": (
        "안내 직원은 {name} 씨에게 방문증을 건넸다.",
        "회의에서 {name}{topic} 지난 분기의 실적을 설명했다.",
        "전화한 사람은 {name}이며 상담 기록을 남겼다.",
        "동료들은 {name}{object} 새 책임자로 추천했다.",
        "민원인 {name}{subject} 창구에서 순서를 기다렸다.",
        "오늘 온 손님의 성함은 {name}이라고 들었다.",
        "{name} 씨는 작성한 서류에 직접 서명했다.",
        "진행자는 참석자 {name} 씨를 소개했다.",
        "직원 {name}{subject} 택배 상자를 받아 갔다.",
        "연구원 {name}{topic} 실험 결과를 발표했다.",
        "보호자인 {name}{subject} 동의 여부를 알려 왔다.",
        "발표가 끝나자 {name} 씨에게 질문이 이어졌다.",
        "신입 {name}{subject} 동료들에게 인사를 건넸다.",
        "서류를 전달한 {name}{topic} 다음 일정으로 이동했다.",
        "참석 여부는 {name} 씨에게 직접 물었다.",
        "동료 {name}{genitive} 의견을 회의록에 반영했다.",
        "신청자는 본인의 이름을 {name}{quotative} 적었다.",
        "손님 {name} 씨, 잠시만 기다려 주세요.",
        "이 서류에 서명한 사람은 {name}이다.",
        "면담 예약을 한 {name}{subject} 시간을 바꾸고 싶다고 말했다.",
    ),
    "validation": (
        "행사 기획자는 연사 {name} 씨의 도착을 확인했다.",
        "심사위원 {name}{subject} 제출된 작품을 읽었다.",
        "소방대원 {name}{topic} 구조 활동을 마치고 복귀했다.",
        "지원자의 추천인은 {name}이라고 기록되어 있다.",
        "조교는 {name} 씨를 실습실로 불렀다.",
        "자문을 맡은 {name}{genitive} 답변을 전달받았다.",
    ),
    "evaluation": (
        "배달 기사는 주문한 고객 {name} 씨와 통화했다.",
        "사회복지사 {name}{subject} 가정 방문 일정을 잡았다.",
        "편집자는 번역가 {name}{object} 저자에게 소개했다.",
        "접수대에서는 헌혈자 {name} 씨의 신분을 확인했다.",
        "공동 구매를 제안한 {name}{topic} 참석자들에게 계좌를 안내했다.",
        "나를 도와준 분의 성함이 {name}이라고 기억한다.",
        "공연을 관람한 {name}{genitive} 후기를 읽었다.",
        "동아리 회장 {name} 씨가 정기 모임을 열었다.",
    ),
}
# The ordinary-word intent is authored separately, rather than inferred from a
# name spelling or a model confidence. These include all-negative business prose.
NEGATIVES = {
    "train": (
        "새해를 맞아 건강과 행복을 기원했다.",
        "조각상에 담긴 미소를 감상했다.",
        "가족의 사랑은 큰 힘이 된다.",
        "서류를 하나만 보내 주세요.",
        "더 이상 문제가 없다고 확인했다.",
        "올해의 소망을 노트에 적었다.",
        "열심히 일한 보람을 느꼈다.",
        "내일에 대한 희망을 품었다.",
        "새봄을 맞아 화단을 정리했다.",
        "가을 하늘이 맑았다.",
        "여름과 겨울의 기온 차이가 크다.",
        "바다 위로 햇빛이 비쳤다.",
        "나무 아래에서 잠시 쉬었다.",
        "봄이 되면 꽃이 핀다.",
        "오늘 처리할 문서는 없습니다.",
        "결제 내역을 다시 확인해 주세요.",
        "신청이 정상적으로 접수되었습니다.",
        "배송 예정일은 다음 주입니다.",
        "회사 상품의 이름을 변경했다.",
        "모두의 건강을 위해 환기했다.",
        "한 개 이상의 파일을 첨부해야 한다.",
        "장비 이상 원인을 조사했다.",
        "미소를 지으며 고개를 끄덕였다.",
        "이웃 사랑을 실천하기로 했다.",
        "회의의 핵심은 비용 절감이다.",
        "계약 내용을 검토한 뒤 회신 바랍니다.",
        "번호와 주소는 별도로 확인한다.",
        "여권과 면허증을 준비해 주세요.",
        "대한민국의 행정구역을 살펴보았다.",
        "시스템에 새 기능을 추가했다.",
    ),
    "validation": (
        "기쁨을 나누면 두 배가 된다고 한다.",
        "평화를 위한 노력이 계속됐다.",
        "노을이 지는 풍경을 바라보았다.",
        "한결 편안한 마음으로 잠들었다.",
        "전자 문서를 저장할 폴더를 만들었다.",
        "재고가 없어 배송이 지연됩니다.",
        "따뜻한 차를 마시며 잠시 쉬었다.",
        "내일도 변함없이 영업합니다.",
    ),
    "evaluation": (
        "여행을 앞둔 설레임에 일찍 잠에서 깼다.",
        "어려움을 슬기롭게 극복했다.",
        "누리 과정에 관한 안내서를 배포했다.",
        "난로의 온기가 방 안에 퍼졌다.",
        "오랜 경험에서 얻은 지혜를 나눴다.",
        "밤하늘의 별을 세어 보았다.",
        "정원에 심은 꽃이 활짝 피어 있었다.",
        "모든 문의는 게시판에 남겨 주세요.",
        "검토할 자료의 양이 예상보다 많았다.",
        "행사 준비에 필요한 장비를 빌렸다.",
        "유지보수 작업으로 접속이 중단됩니다.",
        "연말 정산 안내가 게시되었습니다.",
    ),
}


def particles(name: str) -> dict[str, str]:
    final = ord(name[-1]) - 0xAC00
    consonant = 0 <= final < 11172 and final % 28 != 0
    return dict(
        topic="은" if consonant else "는",
        subject="이" if consonant else "가",
        object="을" if consonant else "를",
        genitive="의",
        quotative="이라고" if consonant else "라고",
    )


def build_cases() -> list[dict]:
    rows = []
    for split, names in NAMES.items():
        for i, template in enumerate(TEMPLATES[split]):
            for j, name in enumerate(names):
                # Build the gold location from the authored placeholder, never
                # locate a possibly repeated name in the rendered sentence.
                prefix, suffix = template.split("{name}")
                text = prefix + name + suffix.format(**particles(name))
                rows.append(
                    dict(
                        id=f"{split}-person-{i:02}-{j:02}",
                        split=split,
                        context_id=f"{split}-person-{i:02}",
                        surface=name,
                        track="prose",
                        text=text,
                        expected=[
                            dict(entity="KR_NAME", start=len(prefix), end=len(prefix) + len(name))
                        ],
                    )
                )
        for i, text in enumerate(NEGATIVES[split]):
            rows.append(
                dict(
                    id=f"{split}-negative-{i:02}",
                    split=split,
                    context_id=f"{split}-negative-{i:02}",
                    surface=None,
                    track="prose",
                    text=text,
                    expected=[],
                )
            )
    # Known errors and the original contrast benchmark become TRAINING data.
    for old in map(json.loads, (ROOT / "data/name_context.jsonl").read_text().splitlines()):
        rows.append(
            dict(
                id="development-" + old["id"],
                split="train",
                context_id="development-" + old["context_id"],
                surface=old["surface"],
                track="prose",
                text=old["text"],
                expected=old["expected"],
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in build_cases())
    if args.check:
        if OUTPUT.read_text(encoding="utf-8") != serialized:
            raise SystemExit("Training corpus differs from authored generator")
    else:
        OUTPUT.write_text(serialized, encoding="utf-8")
    print(f"{len(build_cases())} synthetic cases")


if __name__ == "__main__":
    main()
