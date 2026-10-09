"""Author-controlled synthetic Korean name/context contrasts, without model calls.

This is a developmental challenge, not an NRB reproduction or a population-rate
estimate. Pretraining exposure to the chosen words/names is unknown. Templates
are crossed with surfaces, so sentence counts are not independent sample counts.
The unfamiliar-surname category is constructed, not verified rarity evidence.

Run before inference to freeze the JSONL:
    uv run python benchmarks/name_context_cases.py
Verify it later without rewriting gold:
    uv run python benchmarks/name_context_cases.py --check

Metadata, including unique context/surface counts and hashes, is printed to stdout
and available through corpus_metadata() for evaluators to include in reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

OUTPUT = Path(__file__).parent / "data" / "name_context.jsonl"
FAMILIES = ("ordinary_synthetic", "unfamiliar_surname_synthetic", "homonymous_word")

# Only exact surface spellings define split membership. No frequency, surname
# validity, real identity, or unassigned-name claim is implied by these examples.
SURFACES = {
    "calibration": {
        "ordinary_synthetic": ("김서윤", "이도현", "박민재", "정수빈", "최지우"),
        "unfamiliar_surname_synthetic": ("탁해솔", "견다온", "뇌은찬", "사공이든"),
        "homonymous_word": ("하나", "사랑", "미소"),
    },
    "challenge": {
        "ordinary_synthetic": (
            "김도윤", "박하린", "이서준", "정예린", "최민호", "이상은", "오지은", "김하나",
        ),
        "unfamiliar_surname_synthetic": (
            "빙나래", "즙라온", "어금하람", "독고여울",
            "남궁도담", "황보새봄", "제갈서린", "선우보늬",
        ),
        "homonymous_word": ("이상", "소망", "보람", "희망"),
    },
}

# A context_id is a semantic template family. These families are disjoint
# between calibration and the original challenge. The seen-surface challenge
# deliberately reuses the challenge's person contexts, as a separate experiment.
PERSON_CONTEXTS = {
    "calibration": (
        ("self_introduction", "저는 {name}입니다. 오늘부터 이 팀에서 일합니다."),
        ("signed_application", "신청서의 성명란에는 {name}{quotative} 적혀 있다."),
        ("new_employee_assignment", "신입 직원 {name}{topic} 자료 정리를 맡았다."),
        ("attendance_roll_call", "출석을 확인하던 진행자가 {name} 씨의 이름을 불렀다."),
        ("message_recipient", "담당자에게서 {name} 씨에게 답장을 보내 달라는 요청을 받았다."),
        ("interview_queue", "면접 대상자인 {name}{subject} 다음 순서를 기다렸다."),
        ("parcel_recipient", "배송 안내에 적힌 수령인의 이름은 {name}{past_copula}."),
        ("report_authorship", "이 보고서를 작성한 사람은 {name}이며 공동 작성자는 없다."),
    ),
    "challenge": (
        ("public_announcement", "건물 안내 방송에서 방문객 {name}{object} 찾고 있다고 알렸다."),
        ("clinic_consultation", "간호사는 대기 중인 환자 {name} 씨를 진료실로 안내했다."),
        ("court_testimony", "법정에서 증인 {name}{subject} 당시 상황을 설명했다."),
        ("award_presentation",
         "시상식 사회자는 수상자 {name}{subject} 단상에 올라오도록 요청했다."),
        ("worksite_radio", "안전 점검 도중 작업자 {name} 씨가 무전으로 응답했다."),
        ("family_event", "친척인 {name}{subject} 집안 행사의 준비를 도왔다."),
        ("research_consent", "연구 참여자 {name}{subject} 동의서를 읽고 질문했다."),
        ("classroom_question", "학생 {name}{subject} 손을 들고 질문을 이어 갔다."),
        ("sports_substitution", "감독은 선수 {name}{object} 다음 순서에 출전시키기로 했다."),
        ("stage_preparation", "무대 감독은 배우 {name} 씨의 분장 준비를 확인했다."),
        ("scene_witness", "현장을 목격한 {name}{subject} 담당 조사관과 만났다."),
        ("friend_reference", "그는 친구 {name}{object} 믿을 만한 사람이라고 소개했다."),
        ("wedding_guest", "결혼식 하객인 {name}{subject} 접수대를 지나 자리로 갔다."),
        ("volunteer_duties", "봉사자 {name}{subject} 배식대 옆에서 안내를 맡았다."),
        ("lost_property_owner", "분실물 주인인 {name}{subject} 물건의 특징을 자세히 말했다."),
        ("interpreter_explanation", "통역사 {name}{topic} 질문의 뜻을 다시 설명했다."),
    ),
}

# Explicitly authored ordinary-word uses. These are not model predictions and
# never contain a real person's name hidden in a supposedly negative filename.
NON_PERSON_CONTEXTS = {
    "calibration": {
        "하나": (
            "문서는 하나만 첨부해 주세요.",
            "상자 안에는 연필이 하나 들어 있었다.",
            "두 의견을 하나로 합쳐 정리했다.",
            "남은 선택지는 하나뿐이었다.",
            "준비물 중 하나를 빠뜨렸다.",
            "파일을 하나씩 차례로 열었다.",
            "단추 하나가 바닥에 떨어졌다.",
            "우선 질문 하나를 적어 보자.",
            "예산 항목 하나를 삭제했다.",
            "작은 실수 하나가 일정을 바꾸었다.",
            "필요한 표는 단 하나였다.",
            "여러 조각을 하나로 붙였다.",
        ),
        "사랑": (
            "가족을 향한 사랑이 이야기의 주제다.",
            "서로를 존중하는 사랑을 이야기했다.",
            "이 작품은 부모의 사랑을 다룬다.",
            "동물을 향한 사랑이 봉사의 계기가 되었다.",
            "어려운 시기에도 이웃 사랑을 실천했다.",
            "편지에는 고향에 대한 사랑이 담겼다.",
            "아이에게 사랑을 표현하는 방법을 배웠다.",
            "예술에 대한 사랑이 오랫동안 이어졌다.",
            "우정과 사랑의 차이를 토론했다.",
            "그 시는 조건 없는 사랑을 노래한다.",
            "자연을 향한 사랑은 행동으로 드러났다.",
            "작가는 사랑의 의미를 독자에게 물었다.",
        ),
        "미소": (
            "아이의 얼굴에 미소가 번졌다.",
            "반가운 소식에 저절로 미소가 나왔다.",
            "어색한 순간을 작은 미소로 넘겼다.",
            "사진 속 표정에는 미소가 담겨 있었다.",
            "그는 말없이 미소를 지었다.",
            "친절한 미소가 긴장을 풀어 주었다.",
            "짧은 농담에 모두 미소를 띠었다.",
            "입가의 미소가 오래 남았다.",
            "밝은 미소로 손님을 맞이했다.",
            "그림에서는 인물의 미소가 강조된다.",
            "눈물 사이로 희미한 미소가 보였다.",
            "쑥스러운 미소 뒤에 대답이 이어졌다.",
        ),
    },
    "challenge": {
        "이상": (
            "점검 후 장비에 이상이 없음을 확인했다.",
            "검사 수치의 이상 여부를 다시 살펴보았다.",
            "갑자기 발생한 이상 증상 때문에 기계를 멈췄다.",
            "전원 장치의 이상 원인을 조사했다.",
            "보고서는 발견된 이상 징후를 분류했다.",
            "센서의 이상 작동은 배선 문제에서 비롯됐다.",
            "시스템에 이상이 생기면 경고등이 켜진다.",
            "정기 점검 결과는 이상 없음으로 기록됐다.",
            "소음의 이상 유무를 귀로 확인했다.",
            "평소와 다른 이상 반응이 관찰됐다.",
            "신청 인원이 열 명 이상이면 강좌를 연다.",
            "포장 무게는 기준 이상으로 늘어나지 않았다.",
            "기온이 어제보다 십 도 이상 높았다.",
            "두 개 이상을 동시에 선택할 수 있다.",
            "예산의 절반 이상이 장비 수리에 쓰였다.",
            "출석률이 기준 이상인 경우에만 수료할 수 있다.",
            "수량은 하나 이상이어야 한다.",
            "이상으로 오늘의 설명을 마칩니다.",
            "더 이상 추가할 내용은 없었다.",
            "그 이상은 현재 자료로 판단하기 어렵다.",
            "이상의 내용을 종합해 결론을 내렸다.",
            "현실과 이상의 차이에 관한 글을 읽었다.",
            "공동체가 추구하는 이상을 토론했다.",
            "교육의 이상은 실제 수업과 함께 논의해야 한다.",
            "성능이 기대 이상으로 좋아졌다.",
        ),
        "소망": (
            "새해의 소망은 가족 모두의 건강이다.",
            "학생들은 각자의 소망을 종이에 적었다.",
            "오래 품었던 소망이 마침내 이루어졌다.",
            "여행을 떠나고 싶다는 소망이 생겼다.",
            "작은 소망을 하나씩 실천으로 옮겼다.",
            "편지 마지막에는 평화를 바라는 소망이 담겼다.",
            "많은 사람의 소망을 모아 벽에 붙였다.",
            "새 출발에 대한 소망이 마음을 채웠다.",
            "모두가 무사하기를 바라는 소망은 같았다.",
            "그 글은 더 나은 미래를 향한 소망을 표현한다.",
            "간절한 소망에도 계획은 쉽게 바뀌지 않았다.",
            "자신의 소망을 솔직하게 말해 보았다.",
            "소망의 실현에는 꾸준한 노력이 필요했다.",
            "어린 시절의 소망을 떠올리며 웃었다.",
            "건강하게 지내겠다는 소망을 서로 나누었다.",
            "축하 카드에는 행복을 비는 소망을 적었다.",
            "평범한 일상을 되찾는 것이 가장 큰 소망이었다.",
            "개인의 소망과 공동의 목표를 구분했다.",
            "이야기 속 인물은 마지막 소망을 이루었다.",
            "소망은 있었지만 이를 실행할 시간은 부족했다.",
            "조용히 쉬고 싶다는 소망을 일기에 남겼다.",
            "한 해 동안 이루고 싶은 소망 목록을 만들었다.",
            "서로 다른 소망이 모여 하나의 계획이 되었다.",
            "그 기도에는 이웃의 안녕을 바라는 소망이 있었다.",
            "결과가 어떻든 소망을 잊지 않기로 했다.",
        ),
        "보람": (
            "오랜 연습 끝에 노력의 보람을 느꼈다.",
            "봉사 활동을 마치고 큰 보람이 남았다.",
            "작은 성취에서도 일의 보람을 찾았다.",
            "도움을 받은 사람의 인사가 보람으로 다가왔다.",
            "준비한 시간을 돌아보니 보람이 있었다.",
            "배운 내용을 써 보면서 공부의 보람을 알았다.",
            "힘든 과정이었지만 끝낸 뒤의 보람은 컸다.",
            "수고한 보람이 없다는 말은 하지 않았다.",
            "직원들은 업무에서 느끼는 보람을 이야기했다.",
            "완성된 결과물을 보며 작업의 보람을 느꼈다.",
            "직접 기른 채소를 수확하니 보람이 생겼다.",
            "서로 도운 시간이 보람으로 남았다.",
            "보람의 크기를 보수만으로 설명할 수는 없다.",
            "새로운 기술을 익히는 과정에서 보람을 얻었다.",
            "책 한 권을 끝까지 읽은 보람이 있었다.",
            "행사를 무사히 마쳤다는 사실이 보람이었다.",
            "노력에 비해 보람이 작게 느껴지는 날도 있다.",
            "일상 속 보람을 기록하는 습관을 들였다.",
            "팀의 성장은 지도한 보람을 느끼게 했다.",
            "힘쓴 보람이 드러나기까지 시간이 걸렸다.",
            "보람 없는 경쟁을 계속하고 싶지는 않았다.",
            "문제를 해결한 보람 덕분에 피로를 잊었다.",
            "긴 산행을 마친 뒤 뿌듯한 보람이 남았다.",
            "성공 여부와 별개로 참여한 보람을 찾았다.",
            "그 경험은 보람과 아쉬움을 함께 남겼다.",
        ),
        "희망": (
            "어려운 상황에서도 희망을 잃지 않았다.",
            "회복에 대한 희망이 조금씩 커졌다.",
            "희망 직종을 신청서에 적었다.",
            "희망 근무지는 아직 정하지 않았다.",
            "희망 수량을 확인한 뒤 물품을 주문했다.",
            "희망 날짜를 고르고 예약을 신청했다.",
            "도서관에 희망 도서를 신청했다.",
            "참가 희망 여부를 조사했다.",
            "희망 가격과 실제 가격에는 차이가 있었다.",
            "지원자의 희망 조건을 항목별로 정리했다.",
            "새로운 소식은 모두에게 희망을 주었다.",
            "희망이 있다는 사실만으로도 버틸 수 있었다.",
            "평화를 향한 희망은 쉽게 사라지지 않았다.",
            "절망 속에서도 작은 희망을 발견했다.",
            "미래에 대한 희망을 이야기하는 시간이 마련됐다.",
            "아이들은 희망 가득한 그림을 그렸다.",
            "희망의 불씨를 지키자는 문구를 읽었다.",
            "작품의 주제는 상실 이후에도 남는 희망이다.",
            "기대와 희망을 현실적인 계획으로 옮겼다.",
            "희망 없는 예측만 반복해서는 안 된다.",
            "새로운 치료법에서 희망을 찾는 사람들이 많다.",
            "선택할 수 있는 길이 있다는 것이 희망이었다.",
            "희망 사항과 필수 조건을 구별했다.",
            "교육을 통해 더 나은 삶의 희망을 얻었다.",
            "막연한 희망보다는 구체적인 준비가 필요했다.",
        ),
    },
    "seen_surface_challenge": {
        "하나": (
            "수확한 열매 가운데 상한 것 하나를 골라냈다.",
            "경기 종료 직전에 기회가 하나 더 생겼다.",
            "배터리 하나로 장치를 하루 동안 사용할 수 있다.",
            "복도 끝에는 사용하지 않는 방이 하나 있었다.",
            "실험 조건을 하나 바꾸자 결과가 달라졌다.",
            "길이 갈라지는 곳에서 방향 하나를 선택했다.",
            "어떤 방법 하나가 모든 문제를 해결하지는 못한다.",
            "비상구 하나가 추가로 설치되었다.",
            "완성한 악보에서 음표 하나를 고쳤다.",
            "동전 하나가 의자 밑으로 굴러갔다.",
            "경험 하나를 여러 관점에서 해석했다.",
            "전선 하나를 교체한 뒤 전원이 들어왔다.",
        ),
        "사랑": (
            "철학 수업에서는 사랑과 책임의 관계를 논했다.",
            "정원을 가꾸는 일에서 식물에 대한 사랑이 드러났다.",
            "다큐멘터리는 공동체를 향한 사랑을 보여 주었다.",
            "그 편지는 멀리 떨어진 가족에 대한 사랑의 표현이었다.",
            "노랫말에는 오랜 사랑의 기억이 담겼다.",
            "자신을 향한 사랑도 돌봄의 출발점이 될 수 있다.",
            "헌신과 사랑을 구별하는 질문이 이어졌다.",
            "고향 사랑 기부의 취지를 안내했다.",
            "차이를 인정하는 사랑에 관한 에세이를 읽었다.",
            "동화에서는 사랑이 두려움을 이기는 힘으로 나온다.",
            "함께 보낸 시간이 깊은 사랑으로 이어졌다.",
            "언어마다 사랑을 표현하는 방식이 다르다.",
        ),
        "미소": (
            "긴장이 풀리자 굳었던 얼굴에 미소가 돌아왔다.",
            "조각상의 미소는 보는 방향에 따라 달라 보였다.",
            "손을 흔드는 사람은 환한 미소를 짓고 있었다.",
            "승리를 확인한 순간 벤치에서 미소가 터져 나왔다.",
            "낯선 방문객의 미소에 경계심이 누그러졌다.",
            "무표정하던 배우는 마지막 장면에서 미소를 보였다.",
            "서로 눈이 마주치자 자연스럽게 미소를 나누었다.",
            "입술에 걸린 미소는 잠시 뒤 사라졌다.",
            "기념사진을 찍으며 억지로 미소를 만들지는 않았다.",
            "웃음소리 없이도 미소만으로 반가움을 전할 수 있다.",
            "붓끝으로 인물의 미소를 섬세하게 표현했다.",
            "그 대답을 듣고 안도하는 미소가 번졌다.",
        ),
    },
}

BOUNDARY_SURFACES = ("이상은", "이상", "오지은", "김하나")
BOUNDARY_CONTEXTS = (
    ("boundary_possessive_photo", "증거 사진에는 {name}의 신발이 찍혀 있었다."),
    ("boundary_topic_key", "{name}{topic} 출입문 앞에서 열쇠를 찾고 있었다."),
    ("boundary_subject_umbrella", "버스에서 내린 {name}{subject} 우산을 펼쳤다."),
    ("boundary_object_guard", "경비원은 {name}{object} 안쪽 휴게실로 안내했다."),
    ("boundary_companion_neighbor", "옆집 주민은 {name}{companion} 복도에서 짧게 인사했다."),
    ("boundary_dative_bouquet", "사진 속에는 {name}에게 건넨 꽃다발이 보였다."),
    ("boundary_honorific_entry", "문지기는 {name} 씨의 출입증 사진을 확인했다."),
)


def _surface_rows(split: str) -> list[dict]:
    return [
        {"surface_id": f"{split}:{family}:{index}", "surface": name, "name_family": family}
        for family, names in SURFACES[split].items()
        for index, name in enumerate(names, start=1)
    ]


def _render(template: str, name: str) -> str:
    final_consonant = (ord(name[-1]) - 0xAC00) % 28 != 0
    return template.format(
        name=name, subject="이" if final_consonant else "가",
        topic="은" if final_consonant else "는", object="을" if final_consonant else "를",
        companion="과" if final_consonant else "와",
        past_copula="이었다" if final_consonant else "였다",
        quotative="이라고" if final_consonant else "라고",
    )


def _case(split: str, surface: dict, context_id: str, text: str, track: str) -> dict:
    expected = []
    if track != "non_person_context":
        if text.count(surface["surface"]) != 1:
            raise ValueError("Positive template must contain exactly one name occurrence")
        start = text.index(surface["surface"])
        expected = [{"entity": "KR_NAME", "start": start, "end": start + len(surface["surface"])}]
    elif surface["surface"] not in text:
        raise ValueError("Non-person contrast must contain the target surface")
    return {
        "id": f"{split}:{context_id}:{surface['surface_id']}",
        "group_id": f"{split}:{surface['surface_id']}",
        "context_id": context_id, **surface, "split": split, "track": track,
        "text": text, "expected": expected,
    }


def build_cases() -> list[dict]:
    """Construct fixed gold annotations, with no recognizer or model dependency."""
    cases = []
    for split in ("calibration", "challenge", "seen_surface_challenge"):
        surface_split = "calibration" if split == "seen_surface_challenge" else split
        context_split = "challenge" if split == "seen_surface_challenge" else split
        surfaces = _surface_rows(surface_split)
        for context_id, template in PERSON_CONTEXTS[context_split]:
            for surface in surfaces:
                cases.append(_case(split, surface, context_id,
                                   _render(template, surface["surface"]), "person_context"))
        for surface in surfaces:
            for index, text in enumerate(NON_PERSON_CONTEXTS[split].get(surface["surface"], ())):
                context_id = f"{split}_ordinary_word_{surface['surface']}:{index + 1}"
                cases.append(_case(split, surface, context_id, text, "non_person_context"))
        if split == "challenge":
            for context_id, template in BOUNDARY_CONTEXTS:
                for surface in surfaces:
                    if surface["surface"] in BOUNDARY_SURFACES:
                        cases.append(_case(split, surface, context_id,
                                           _render(template, surface["surface"]), "boundary"))
    return cases


def serialize_cases(cases: list[dict]) -> str:
    return "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases)


def _counts(cases: list[dict]) -> dict:
    polarities_by_group: dict[str, set[bool]] = {}
    for case in cases:
        polarities_by_group.setdefault(case["group_id"], set()).add(bool(case["expected"]))
    return {
        "sentences": len(cases),
        "positive_sentences": sum(bool(case["expected"]) for case in cases),
        "negative_sentences": sum(not case["expected"] for case in cases),
        "unique_surfaces": len({case["surface"] for case in cases}),
        "unique_contexts": len({case["context_id"] for case in cases}),
        "surface_groups": len(polarities_by_group),
        "paired_positive_negative_groups": sum(
            len(polarities) == 2 for polarities in polarities_by_group.values()
        ),
        "by_track": dict(sorted(Counter(case["track"] for case in cases).items())),
        "by_name_family": dict(sorted(Counter(case["name_family"] for case in cases).items())),
    }


def corpus_metadata(cases: list[dict] | None = None) -> dict:
    cases = build_cases() if cases is None else cases
    return {
        "schema_version": 1,
        "corpus_kind": "synthetic developmental challenge; author-controlled labels",
        "pretraining_exposure": "unknown",
        "nrb_reproduction": False,
        "population_rate_estimate": False,
        "independence_warning": (
            "Templates are crossed with surfaces. Sentence counts are not independent samples. "
            "Report surface groups, paired polarity groups, and unique contexts alongside counts."
        ),
        "split_design": {
            "calibration": "Authored calibration surfaces and semantic context families.",
            "challenge": (
                "Target surfaces and context families disjoint from calibration. "
                "Background tokens may include calibration words; this is not token disjointness."
            ),
            "seen_surface_challenge": (
                "Calibration surfaces in challenge person contexts plus newly authored "
                "ordinary-word contexts; intentional overlap, report separately."
            ),
        },
        "name_family_note": (
            "All names are synthetic examples; unfamiliar surname spellings are not verified rare. "
            "Examples may coincide with real names but describe no collected personal records."
        ),
        **_counts(cases),
        "per_split": {
            split: _counts([case for case in cases if case["split"] == split])
            for split in ("calibration", "challenge", "seen_surface_challenge")
        },
        "data_sha256": hashlib.sha256(serialize_cases(cases).encode("utf-8")).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--check", action="store_true", help="Verify frozen corpus without rewriting",
    )
    args = parser.parse_args()
    cases = build_cases()
    encoded = serialize_cases(cases)
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != encoded:
            raise SystemExit("Frozen corpus differs from the authored generator")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps(corpus_metadata(cases), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
