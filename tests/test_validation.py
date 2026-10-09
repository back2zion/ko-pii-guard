import pytest

from ko_pii_guard.validation import has_valid_registration_date


@pytest.mark.parametrize("entity,number", [
    ("KR_RRN", "900231-1234567"),
    ("KR_RRN", "990431-2234567"),
    ("KR_RRN", "000229-1234567"),  # 1900 was not a leap year.
    ("KR_RRN", "000229-9234567"),  # 1800 was not a leap year either.
    ("KR_RRN", "230229-3234567"),
    ("KR_RRN", "240600-3234567"),
    ("KR_RRN", "241301-4234567"),
    ("KR_FRN", "000229-5234567"),
    ("KR_FRN", "230229-7234567"),
    ("KR_FRN", "990631-6234567"),
    ("KR_FRN", "241131-8234567"),
])
def test_impossible_calendar_dates(entity, number):
    assert not has_valid_registration_date(entity, number)
    assert not has_valid_registration_date(entity, number.replace("-", ""))


@pytest.mark.parametrize("entity,number", [
    ("KR_RRN", "960229-1234567"),
    ("KR_RRN", "960229-2234567"),
    ("KR_RRN", "000229-3234567"),  # 2000 was a leap year.
    ("KR_RRN", "240229-4234567"),
    ("KR_RRN", "960229-9234567"),  # 1896 was a leap year.
    ("KR_RRN", "960229-0234567"),
    ("KR_FRN", "960229-5234567"),
    ("KR_FRN", "960229-6234567"),
    ("KR_FRN", "000229-7234567"),
    ("KR_FRN", "240229-8234567"),
    ("KR_RRN", "991231-3234567"),  # Calendar validity is independent of current date.
])
def test_supported_century_codes_and_leap_days(entity, number):
    assert has_valid_registration_date(entity, number)
    assert has_valid_registration_date(entity, number.replace("-", ""))


@pytest.mark.parametrize("entity,code", [("KR_RRN", "3"), ("KR_FRN", "7")])
def test_modern_random_suffix_does_not_need_legacy_checksum(entity, code):
    # All ten check-digit possibilities remain candidates; this is not a
    # checksum or real-identifier validity API.
    for check_digit in range(10):
        assert has_valid_registration_date(entity, f"240229-{code}99999{check_digit}")


@pytest.mark.parametrize("entity,number", [
    ("KR_RRN", "900101-5234567"),
    ("KR_FRN", "900101-1234567"),
    ("KR_RRN", "900101--1234567"),
    ("KR_RRN", "900101 1234567"),
    ("KR_RRN", "900101-123456"),
    ("KR_RRN", "900101-12345678"),
    ("KR_FRN", "900101-523456A"),
    ("KR_RRN", "900101-1234567\n"),
    ("KR_RRN", ""),
])
def test_only_normalized_complete_registration_candidates_are_checked(entity, number):
    assert not has_valid_registration_date(entity, number)


@pytest.mark.parametrize("entity,span", [
    ("KR_ACCOUNT", "9002311234567"),
    ("PHONE_NUMBER", "010-1234-5678"),
    ("EMAIL_ADDRESS", "person@example.com"),
    ("UNKNOWN_ENTITY", ""),
])
def test_other_entity_types_pass_through(entity, span):
    assert has_valid_registration_date(entity, span)
