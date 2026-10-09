"""Documented account *layouts*, not an account or bank identity validator.

Sources only cover the products and publication dates described in each entry.
The patterns have no text boundaries: callers must check complete candidate
boundaries, local context, and conflicts with other identifiers themselves.
See ``docs/account-format-sources.md`` for provenance and coverage limitations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FSS_SOURCE = "https://eiec.kdi.re.kr/policy/materialView.do?num=249852"
SEMAS_SOURCE = "https://ols.semas.or.kr/ols/downloads/ols_guide_18.pdf"
KAKAO_SOURCE = "https://eng.kakaobank.com/products/group-account"


@dataclass(frozen=True, slots=True)
class AccountFormat:
    """An immutable, source-backed candidate layout using ASCII digits."""

    name: str
    bank: str
    regex: str
    group_count: int
    source_url: str
    applicability: str
    compiled: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "compiled", re.compile(self.regex))


ACCOUNT_FORMATS: tuple[AccountFormat, ...] = (
    AccountFormat(
        "nh_current", "농협", r"(?:301|302|312)-[0-9]{4}-[0-9]{4}-[0-9]{2}", 4,
        SEMAS_SOURCE, "NH농협은행 신계좌: 소상공인정책자금 이용안내 2022-12 사본, 33쪽",
    ),
    AccountFormat(
        "nh_local_current", "농협", r"(?:351|352|356)-[0-9]{4}-[0-9]{4}-[0-9]{2}", 4,
        SEMAS_SOURCE, "지역농협 신계좌: 소상공인정책자금 이용안내 2022-12 사본, 33쪽",
    ),
    AccountFormat(
        "nh_legacy", "농협", r"[0-9]{3}-(?:01|02|12)-[0-9]{6}", 3,
        SEMAS_SOURCE, "NH농협은행 구계좌: 소상공인정책자금 이용안내 2022-12 사본, 33쪽",
    ),
    AccountFormat(
        "nh_local_legacy", "농협", r"[0-9]{6}-(?:51|52|56)-[0-9]{6}", 3,
        SEMAS_SOURCE, "지역농협 구계좌: 소상공인정책자금 이용안내 2022-12 사본, 33쪽",
    ),
    AccountFormat(
        "kb_savings", "국민", r"[0-9]{4}(?:03|23|26)-[0-9]{2}-[0-9]{6}", 3,
        FSS_SOURCE, "2024-02 은행 제출 적금계좌 체계, 금융감독원 자료 5쪽",
    ),
    AccountFormat(
        "shinhan_savings", "신한", r"(?:230|223)-[0-9]{3}-[0-9]{6}", 3,
        FSS_SOURCE, "2024-02 은행 제출 적금계좌 체계, 금융감독원 자료 5쪽",
    ),
    AccountFormat(
        "hana_savings", "하나", r"[0-9]{3}-[0-9]{6}-[0-9]{3}(?:21|25)", 3,
        FSS_SOURCE, "2024-02 은행 제출 적금계좌 체계, 금융감독원 자료 5쪽",
    ),
    AccountFormat(
        "suhyup_savings", "수협", r"(?:1400|1410)-[0-9]{4}-[0-9]{4}", 3,
        FSS_SOURCE, "2024-02 은행 제출 적금계좌 체계, 금융감독원 자료 5쪽",
    ),
    AccountFormat(
        "ibk_savings", "기업", r"[0-9]{3}-[0-9]{6}-14-[0-9]{3}", 4,
        FSS_SOURCE, "2024-02 은행 제출 적금계좌 체계, 금융감독원 자료 5쪽",
    ),
    AccountFormat(
        "kdb_savings", "산업", r"(?:031|032|037)-[0-9]{4}-[0-9]{4}-[0-9]{3}", 4,
        FSS_SOURCE, "2024-02 은행 제출 적금계좌 체계, 금융감독원 자료 6쪽",
    ),
    AccountFormat(
        "kakao_checking", "카카오뱅크", r"3333-[0-9]{2}-[0-9]{7}", 3,
        KAKAO_SOURCE, "공식 모임통장 상품 페이지의 입출금통장 표시 예시; 전체 상품 체계 아님",
    ),
    AccountFormat(
        "kakao_savings", "카카오뱅크", r"[0-9]355-[0-9]{2}-[0-9]{7}", 3,
        FSS_SOURCE, "2024-02 은행 제출 적금계좌 체계, 금융감독원 자료 6쪽",
    ),
    AccountFormat(
        "kbank_savings", "케이뱅크", r"110-2[0-9]{2}-[0-9]{6}", 3,
        FSS_SOURCE, "2024-02 은행 제출 적금계좌 체계, 금융감독원 자료 6쪽",
    ),
    AccountFormat(
        "toss_savings", "토스뱅크", r"300[0-9]-[0-9]{4}-[0-9]{4}", 3,
        FSS_SOURCE, "2024-02 은행 제출 적금계좌 체계, 금융감독원 자료 6쪽",
    ),
    AccountFormat(
        "busan_savings", "부산", r"104-[0-9]{4}-[0-9]{4}-[0-9]{2}", 4,
        FSS_SOURCE, "2024-02 은행 제출 적금계좌 체계, 금융감독원 자료 6쪽",
    ),
    AccountFormat(
        "citi_savings_short", "씨티", r"[0-9]{3}-[0-9]{5}-(?:16|18|19|20|37|38|39)[0-9]", 3,
        FSS_SOURCE, "2024-02 은행 제출 적금계좌 체계 중 11자리, 금융감독원 자료 6쪽",
    ),
    AccountFormat(
        "citi_savings_long", "씨티",
        r"[0-9]{3}-[0-9]{5}-(?:16|18|19|20|37|38|39)[0-9]-[0-9]{2}", 4,
        FSS_SOURCE, "2024-02 은행 제출 적금계좌 체계 중 13자리, 금융감독원 자료 6쪽",
    ),
)

FOUR_GROUP_ACCOUNT_FORMATS: tuple[AccountFormat, ...] = tuple(
    layout for layout in ACCOUNT_FORMATS if layout.group_count == 4
)


def matches_account_format(value: str) -> bool:
    """Whether a complete hyphenated candidate has a documented layout.

    A match is structural evidence only. It does not establish that the value is
    PII, identify its bank, or confirm that an account exists. An unknown layout
    may still be a real account. This function does not normalize Unicode.
    """
    return any(layout.compiled.fullmatch(value) for layout in ACCOUNT_FORMATS)
