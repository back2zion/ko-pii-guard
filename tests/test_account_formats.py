"""Independent examples of documented layouts and confusing near-misses.

Numbers below were constructed for tests, not copied from account holders.
They have no verified assignment status and must not be used for transactions.
"""

import pytest

from ko_pii_guard.account_formats import ACCOUNT_FORMATS, matches_account_format


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("nh_current", "302-8642-9753-10"),
        ("nh_local_current", "352-8642-9753-10"),
        ("nh_legacy", "864-02-975310"),
        ("nh_local_legacy", "864297-52-531086"),
        ("kb_savings", "864203-97-531086"),
        ("shinhan_savings", "230-864-975310"),
        ("hana_savings", "864-297531-08621"),
        ("suhyup_savings", "1400-8642-9753"),
        ("ibk_savings", "864-297531-14-086"),
        ("kdb_savings", "031-8642-9753-108"),
        ("kakao_checking", "3333-86-4297531"),
        ("kakao_savings", "3355-86-4297531"),
        ("kbank_savings", "110-286-429753"),
        ("toss_savings", "3008-6429-7531"),
        ("busan_savings", "104-8642-9753-10"),
        ("citi_savings_short", "864-29753-168"),
        ("citi_savings_long", "864-29753-168-10"),
    ],
)
def test_documented_layouts(name, value):
    layout = next(layout for layout in ACCOUNT_FORMATS if layout.name == name)
    assert layout.compiled.fullmatch(value)
    assert matches_account_format(value)
    assert value.count("-") + 1 == layout.group_count


@pytest.mark.parametrize(
    "value",
    [
        "2026-10-09",  # Date.
        "010-8642-9753",  # Telephone number.
        "11-26-864297-53",  # Driver's license layout.
        "4000-8642-9753-1086",  # Payment-card layout.
        "302-8642-9753",  # The old generator omitted the last NH group.
        "303-8642-9753-10",  # Do not extrapolate a 3xx prefix from NH examples.
        "864-297531-15-086",  # Only the documented IBK product code is represented.
        "031-8642-9753-10",  # KDB has a three-digit final group in the source.
        "104-8642-9753-108",  # Busan has a two-digit final group in the source.
        "864-29753-178-10",  # Unlisted Citi product code.
        "3028642975310",  # Caller must choose undelimited detection separately.
        "302-8642-9753-10-20",  # Full match must not silently accept a fragment.
        "x302-8642-9753-10",
        "302-8642-9753-10x",
        "３０２-８６４２-９７５３-１０",  # Normalization is the caller's responsibility.
    ],
)
def test_similar_values_are_not_documented_layouts(value):
    assert not matches_account_format(value)
