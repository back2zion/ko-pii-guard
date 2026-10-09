"""Conservative Korean road-name and land-lot address recognition.

Recognize address *syntax*, not an address's existence or deliverability. Require
either a known province/metro name and a local administrative chain, or an
immediately preceding address label. Bare location names are deliberately absent.
No national road/locality dictionary is bundled; no network requests are made.

Primary references reviewed 2026-10-10:
* Road/lot fields and road-name notation examples:
  https://www.archives.go.kr/next/pages/common/newAddressJibun2.jsp
* Juso building numbers and attached building/floor/unit detail fields:
  https://business.juso.go.kr/addrlink/attrbDBDwld/attrbDBDwldList.do?cPath=99MD
* Full province/metro names (current and historical aliases retained below):
  https://jumin.mois.go.kr/statMonth.do
* July 2026 Jeonnam/Gwangju merger commencement:
  https://mois.go.kr/frt/bbs/type010/commonSelectBoardArticle.do?bbsId=BBSMSTR_000000000008&nttId=126845

Separated Korean road/locality tokens and house/lot numbers are supported, with
at most one line break per token boundary. Numeric 동/층/호 details and bounded
parenthetical locality/building reference fields may follow. Free-form delivery
comments, unspaced addresses and prose-only locations are outside this grammar.
Aliases are hand-maintained spelling facts, not a copied address DB.
"""

from __future__ import annotations

import re

from presidio_analyzer import RecognizerResult

# Current names plus familiar abbreviations and legacy administrative names.
# These aliases do not purport to enumerate only today's administrative units.
PROVINCE_ALIASES = (
    "서울특별시", "서울시", "서울", "부산광역시", "부산시", "부산",
    "대구광역시", "대구시", "대구", "인천광역시", "인천시", "인천",
    "광주광역시", "광주시", "광주", "대전광역시", "대전시", "대전",
    "울산광역시", "울산시", "울산", "세종특별자치시", "세종시", "세종",
    "경기도", "경기", "강원특별자치도", "강원도", "강원",
    "충청북도", "충북", "충청남도", "충남", "전북특별자치도", "전라북도", "전북",
    "전라남도", "전남", "경상북도", "경북", "경상남도", "경남",
    "제주특별자치도", "제주도", "제주", "전남광주통합특별시",
)

_GAP = r"[ \t]{0,16}"
_LINEBREAK = rf"{_GAP}\r?\n{_GAP}"
_WS = rf"(?:[ \t]{{1,16}}|{_LINEBREAK})"
_PROVINCE = "(?:" + "|".join(sorted(PROVINCE_ALIASES, key=len, reverse=True)) + ")"
_ADMIN = r"[가-힣]{1,12}[시군구]"
_RURAL = rf"(?:[가-힣0-9·]{{1,12}}[읍면]{_WS})?"
_ROAD = r"[가-힣0-9·.]{1,30}(?:대로|로|길)"
_LOCALITY = r"[가-힣0-9·]{1,15}(?:동|리|가)"
_NUMBER = r"[1-9][0-9]{0,4}(?:-[0-9]{1,4})?"
_STREET = rf"{_ROAD}{_WS}(?:지하{_GAP})?{_NUMBER}"
_LOT = rf"{_LOCALITY}{_WS}(?:산{_GAP})?{_NUMBER}(?:번지)?"
_BASE = rf"{_RURAL}(?:{_STREET}|{_LOT})"

_BUILDING = r"[가-힣A-Za-z0-9]{1,24}(?:아파트|빌라|오피스텔|빌딩)"
_DETAIL_TOKEN = (
    rf"(?:[0-9]{{1,5}}동|[A-Z]동|(?:지하{_GAP}|B)?[0-9]{{1,4}}층|[0-9]{{1,5}}호)"
)
_DETAIL = (
    rf"(?:{_GAP},{_GAP}|{_WS})(?:{_BUILDING}{_WS})?"
    rf"{_DETAIL_TOKEN}(?:(?:[ \t,]{{0,16}}|{_LINEBREAK}){_DETAIL_TOKEN}){{0,2}}"
)
# Juso reference fields are a legal locality and/or a building name. An
# arbitrary bounded building token is allowed only after a locality and comma;
# standalone notes need an explicit building-type suffix. This excludes prose
# instructions such as '(문앞에 놓아주세요)' without a building-name dictionary.
_NOTE_BUILDING = r"[가-힣A-Za-z0-9·]{1,30}"
_NOTE = (
    rf"(?:{_GAP}|{_LINEBREAK})\({_GAP}(?:"
    rf"{_LOCALITY}(?:{_GAP},{_GAP}{_NOTE_BUILDING})?|{_BUILDING}){_GAP}\)"
)
_SUFFIX = rf"(?:{_DETAIL}(?:{_NOTE})?|{_NOTE}(?:{_DETAIL})?)?"
_BOUNDARY = r"(?<![가-힣A-Za-z0-9_])"
_END = r"(?![A-Za-z0-9_-])"

_FULL = re.compile(
    rf"{_BOUNDARY}(?P<address>(?:"
    rf"{_PROVINCE}{_WS}(?:{_ADMIN}{_WS}){{1,2}}"
    rf"|(?:세종특별자치시|세종시|세종){_WS}){_BASE}{_SUFFIX}){_END}"
)
_LABELED = re.compile(
    rf"{_BOUNDARY}(?:도로명주소|지번주소|배송주소|자택주소|직장주소|주소|배송지|거주지)"
    rf"(?:는|은)?[\"']?{_GAP}[:：=]?{_GAP}(?:\r?\n{_GAP})?[\"']?"
    rf"(?P<address>(?:{_PROVINCE}{_WS})?(?:{_ADMIN}{_WS}){{0,2}}{_BASE}{_SUFFIX}){_END}"
)
_PARTICLE = re.compile(r"(?:입니다|이고|이며|이었다|였다|에서|으로|까지|부터|로|에|은|는|을|를)")


def recognize_addresses(text: str) -> list[RecognizerResult]:
    """Return conservative KR_ADDRESS candidates, with offsets into ``text``.

    Address field labels are excluded from returned spans. Results are not
    evidence that an address exists or that the resident is a specific person.
    """
    candidates: dict[tuple[int, int], RecognizerResult] = {}
    for pattern, score in ((_FULL, 0.65), (_LABELED, 0.7)):
        for match in pattern.finditer(text):
            start, end = match.span("address")
            # Avoid reading a quantity such as '테헤란로 3개' or '출근길 20분'
            # as a house number. Familiar sentence particles can follow it.
            if end < len(text) and "가" <= text[end] <= "힣" and not _PARTICLE.match(text, end):
                continue
            candidates[start, end] = RecognizerResult(
                entity_type="KR_ADDRESS", start=start, end=end, score=score,
                recognition_metadata={
                    RecognizerResult.RECOGNIZER_NAME_KEY: "KrAddressRecognizer",
                    RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: "KrAddressRecognizer",
                },
            )
    return sorted(candidates.values(), key=lambda result: (result.start, -result.end))
