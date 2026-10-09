"""Conservative person-name fields; free prose is handled by optional local NER.

There is no surname allow-list and no generic two/three-syllable name rule.
Explicit fields can contain rare surnames, transliterations and Latin names.
"""

from __future__ import annotations

import re
import unicodedata

import regex
from presidio_analyzer import RecognizerResult

_LABEL = re.compile(
    r"(?<![가-힣A-Za-z0-9_])"
    r"(?:성명|성함|이름|고객명|담당자|예금주|수취인|수령인|받는[ \t]*분|신청인"
    r"|수신자|발신인|책임자|참고인|수신|발신)"
    r"[\"']?[ \t]*(?:[:=：][ \t]*(?:\r?\n[ \t]*)?|[ \t]+)[\"']?"
)
_VALUE = regex.compile(
    r"\p{L}[\p{L}\p{M}'’\-]{0,24}"
    r"(?:[ \t]+\p{L}[\p{L}\p{M}'’\-]{0,24}){0,3}"
)
_NOT_NAMES = frozenset((
    "없음", "미기재", "미입력", "비공개", "익명", "알수없음", "해당없음", "테스트",
    "고객센터", "담당자", "관리자", "이름", "성명", "회사", "회사명", "주식회사",
    "성함", "수취인", "수령인", "예금주", "성명란", "이름란",
    "null", "none", "unknown", "n/a",
))
_NON_PERSON_OWNER = re.compile(r"(?:상품|제품|파일|프로그램|문서|회사|법인|서비스|변수)[ \t]*$")
_NEXT_FIELD = re.compile(
    r"[ \t]+(?:성명|성함|이름|고객명|담당자|예금주|수취인|수령인|연락처|전화번호|주소|이메일"
    r"|수신자|발신인|책임자|참고인|수신|발신)"
    r"[ \t]*[:=：]"
)
_ACTION = r"(?:확인|입력|변경|조회|등록|수정|검색|작성|기재|선택|삭제|수집|지정|선임)"
_INSTRUCTION = re.compile(
    _ACTION + r"(?:(?:이|가|은|는)?[ \t]+(?:완료|필요|요청|부탁|신청|후|중)|해|하)"
    r"|(?:부탁드립니다|해주세요|바랍니다|하십시오|하세요)"
)
_BARE_ACTION = re.compile(_ACTION + r"(?:[ \t]|$)")
_NON_PERSON_FIELD = re.compile(
    r"(?:회사명|법인명|상호|상품명|제품명|프로젝트명|서비스명|변수명|주소|배송지)"
    r"[\"']?[ \t]*[:=：][ \t]*[\"']?$"
)
_NAMED_OBJECT = re.compile(
    r"(?:이?라는|이?란|이라는 이름의|이라는 제목의)[ \t]+"
    r"(?:상품|제품|프로젝트|회사|변수|함수|서비스|문서|책|영화|노래)"
)
_ROLE_FIELD = re.compile(
    r"(?:담당자|수령인|수취인|신청인|수신자|발신인|책임자|참고인|수신|발신)"
    r"[\"']?[ \t]*[:=：]?[ \t]*[\"']?$"
)
_ROLE_PLACEHOLDERS = frozenset(("미정", "미지정", "선임예정", "법인", "관리실",
                                "관계자", "관계자여러분", "전직원", "귀하", "담당부서", "자동발송"))
_ROLE_ORGANIZATION = re.compile(r"[가-힣]{2,}(?:팀|부서|본부|위원회)$")


def _role_placeholder(value: str) -> bool:
    compact = "".join(value.split())
    return compact in _ROLE_PLACEHOLDERS or bool(_ROLE_ORGANIZATION.fullmatch(compact))


def allows_contextual_name(text: str, start: int, end: int) -> bool:
    """Veto explicit non-person semantics, not unfamiliar name spellings.

    A model's high confidence does not override an explicit product/address
    field or turn a missing-value/template marker into a person's name.
    """
    value = text[start:end]
    if (not any(c.isalpha() for c in value)
            or any(unicodedata.category(c)[0] not in "LM" and c not in " .·'’-\t"
                   for c in value)
            or value.lower() in _NOT_NAMES or _INSTRUCTION.search(value)):
        return False
    if _NON_PERSON_FIELD.search(text[max(0, start - 40):start]):
        return False
    if _role_placeholder(value) and _ROLE_FIELD.search(text[max(0, start - 40):start]):
        return False
    # The classifier may include the copula in its span. Inspect the candidate
    # and a bounded right context, instead of assuming its boundary is correct.
    relation = _NAMED_OBJECT.search(text, start, min(len(text), end + 24))
    if relation is not None and relation.start() <= end:
        return False
    return True


def recognize_name_fields(text: str) -> list[RecognizerResult]:
    results = []
    for label in _LABEL.finditer(text):
        if _NON_PERSON_OWNER.search(text[max(0, label.start() - 15):label.start()]):
            continue
        # A bare role word in prose is not a structured field. Require a
        # delimiter or a line-leading label; let NER handle prose names.
        if not any(ch in label.group() for ch in ":=："):
            prefix = text[text.rfind("\n", 0, label.start()) + 1:label.start()].strip()
            if prefix not in ("", "-", "*", "•", ">"):
                continue
        boundary = _NEXT_FIELD.search(text, label.end(), label.end() + 120)
        match = _VALUE.match(text, label.end(), boundary.start() if boundary else len(text))
        if match is None:
            continue
        value = match.group()
        # Respect explicit field boundaries. Do not grab the beginning of a
        # number, organization, identifier, or longer free-form sentence.
        end = match.end()
        if end < len(text) and (text[end].isalnum() or text[end] in "_-@"):
            continue
        if (_INSTRUCTION.search(value)
                or (not any(ch in label.group() for ch in ":=：")
                    and _BARE_ACTION.match(value))):
            continue
        honorific = next((suffix for suffix in (" 님", " 씨", " 귀하")
                          if value.endswith(suffix)), None)
        if honorific:
            value = value[:-len(honorific)]
            end -= len(honorific)
        elif len(value) >= 4 and value.endswith(("님", "씨")):
            value = value[:-1]
            end -= 1
        if (len(value.replace(" ", "")) < 2 or value.lower() in _NOT_NAMES
                or any(part.lower() in _NOT_NAMES for part in value.split())
                or value.endswith(("회사", "은행", "센터", "대학교", "병원", "입니다"))):
            continue
        if _role_placeholder(value) and _ROLE_FIELD.search(label.group()):
            continue
        results.append(RecognizerResult("KR_NAME", match.start(), end, 0.85))
    return results
