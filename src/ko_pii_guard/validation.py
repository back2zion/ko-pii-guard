"""Calendar checks for Korean registration-number candidates.

Century codes are documented in Statistics Korea's 2010 second-half research
report, volume II, printed page 200 (PDF page 18):
https://kostat.go.kr/boardDownload.es?bid=11887&list_no=369713&seq=1
The 1900/2000 codes are also explained by Korea's policy briefing service:
https://www.korea.kr/briefing/policyBriefingView.do?newsId=148818721

This checks a candidate's date, not whether an identifier was issued. In
particular, it does not require a legacy checksum or region code. Callers decide
whether explicit sensitive-field context should retain a mistyped candidate.
"""

from __future__ import annotations

import re
from datetime import date

_REGISTRATION_PATTERN = re.compile(r"[0-9]{6}-?[0-9]{7}\Z")
_CENTURIES = {
    "KR_RRN": {"0": 1800, "1": 1900, "2": 1900, "3": 2000, "4": 2000, "9": 1800},
    "KR_FRN": {"5": 1900, "6": 1900, "7": 2000, "8": 2000},
}


def has_valid_registration_date(entity: str, span: str) -> bool:
    """Check the birth date in an already normalized RRN/FRN candidate.

    Other entity types return True so this can be composed with unrelated
    candidate filters. RRN/FRN inputs must contain exactly 13 ASCII digits with
    an optional hyphen after the birth date. No checksum, region, serial-number,
    age or date-of-issue restriction is imposed.

    A False result is evidence of an impossible date or incompatible format,
    not permission to expose a value explicitly labelled as personal data.
    """
    centuries = _CENTURIES.get(entity)
    if centuries is None:
        return True
    if not _REGISTRATION_PATTERN.fullmatch(span):
        return False
    digits = span.replace("-", "")
    century = centuries.get(digits[6])
    if century is None:
        return False
    try:
        date(century + int(digits[:2]), int(digits[2:4]), int(digits[4:6]))
    except ValueError:
        return False
    return True
