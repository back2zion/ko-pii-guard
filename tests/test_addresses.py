"""Address grammar tests use illustrative strings, not private address records."""

import json
from pathlib import Path

import pytest

from ko_pii_guard.addresses import recognize_addresses


@pytest.mark.parametrize("address", [
    "서울특별시 강남구 가상누리로 9999",
    "서울 강남구 가상누리대로 999-1",
    "부산광역시 해운대구 가상누리로12길 99-1",
    "경기도 수원시 팔달구 가상동 999-1",
    "경기 성남시 분당구 가상누리로 999",
    "충남 홍성군 홍북읍 가상리 999-1번지",
    "경상북도 안동시 가상면 가상리 산 99-1",
    "강원특별자치도 춘천시 가상누리길 999",
    "전북특별자치도 전주시 완산구 가상누리로 999",
    "제주특별자치도 제주시 가상읍 가상누리로 999",
    "세종특별자치시 가상누리로 9999",
    "세종시 조치원읍 가상리 999",
    "전남광주통합특별시 목포시 가상누리로 999",
    "광주광역시 서구 가상누리로 999",
    "전라남도 목포시 가상누리로 999",
])
def test_full_administrative_address(address):
    text = f"보내실 곳은 {address} 입니다"
    results = recognize_addresses(text)
    assert len(results) == 1
    result = results[0]
    assert result.entity_type == "KR_ADDRESS"
    assert text[result.start:result.end] == address


@pytest.mark.parametrize("label", ["주소:", "주소는", "배송지:", "거주지:", "도로명주소:"])
@pytest.mark.parametrize("address", [
    "가상누리로 9999", "가상동 999-1", "강남구 가상누리길 999",
    "수원시 팔달구 가상동 999-1", "서울특별시 강남구 가상누리로 9999",
])
def test_explicit_label_allows_shorter_address(label, address):
    text = f"{label} {address} 확인"
    assert {text[r.start:r.end] for r in recognize_addresses(text)} == {address}


@pytest.mark.parametrize("detail", [
    " 101동 202호", ", 101동 202호", " 101동202호", " 지하 1층 101호",
    ", 테스트아파트 101동 202호", " A동 3층 301호", " B1층",
])
def test_attached_address_details(detail):
    address = f"서울특별시 강남구 가상누리로 9999{detail}"
    text = f"주소: {address} 확인"
    assert {text[r.start:r.end] for r in recognize_addresses(text)} == {address}


@pytest.mark.parametrize("text", [
    "서울특별시", "서울 강남구", "서울 강남구 가상누리로", "가상누리로 9999",
    "가상동 999-1", "서울 강남구 가상동", "주소: 미기재", "주소록 9999",
    "서버 주소: localhost", "메모리 주소: 0xABCD", "배송지 미정",
    "서울 강남구 도보로 3분", "서울 강남구 출근길 20분", "주소: 가상누리로 3개",
    "주소: 가상누리로 9999ABC", "주소: 가상누리로 123456",
    "주소: 가상누리로 010-1234-5678", "주소: 가상누리로 2026-10-09",
    "주소: 가상동 123-456-789", "주소 안내; 가상동 123",
    "주소: 가상누리로\n\n9999", "주소: 가상동 999층",
])
def test_ambiguous_places_codes_and_numbers_are_not_addresses(text):
    assert recognize_addresses(text) == []


def test_address_does_not_consume_following_contact():
    address = "서울특별시 강남구 가상누리로 9999 101동 202호"
    text = f"주소: {address}, 연락처 010-1234-5678"
    assert {text[r.start:r.end] for r in recognize_addresses(text)} == {address}


def test_particle_after_address_is_not_part_of_span():
    address = "서울 강남구 가상누리로 9999"
    text = f"{address}로 보내 주세요"
    assert [text[r.start:r.end] for r in recognize_addresses(text)] == [address]


def test_multiple_addresses_keep_individual_spans():
    first = "서울 강남구 가상누리로 9999"
    second = "부산 해운대구 가상누리길 8888"
    text = f"출발 {first}; 도착 {second}"
    assert [text[r.start:r.end] for r in recognize_addresses(text)] == [first, second]


@pytest.mark.parametrize("template", ['{{"주소": "{}"}}', "주소:\n{}", "주소={}"])
def test_structured_address_field_delimiters(template):
    address = "가상누리로 9999"
    text = template.format(address)
    assert [text[r.start:r.end] for r in recognize_addresses(text)] == [address]


def test_structured_address_fixture_spans():
    path = Path(__file__).resolve().parents[1] / "benchmarks/data/name_address.jsonl"
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["track"] != "structured":
            continue
        expected = sorted(
            (span["start"], span["end"]) for span in row["expected"]
            if span["entity"] == "KR_ADDRESS"
        )
        actual = sorted((span.start, span.end) for span in recognize_addresses(row["text"]))
        assert actual == expected, row["id"]


def test_long_whitespace_or_tokens_do_not_create_address_fragments():
    assert recognize_addresses("주소:" + " " * 10_000 + "가상누리로 9999") == []
    assert recognize_addresses("서울 강남구 " + "가" * 10_000 + "로 9999") == []


@pytest.mark.parametrize("suffix", [
    " (가상동)", "(가상동, 테스트아파트)", " (가상동, 가상누리자이)",
    " (테스트아파트)", " (가상동, 테스트아파트) 101동 202호",
    ", 101동 202호 (가상동, 테스트아파트)",
    " (가상동, 테스트아파트)\n101동\n202호",
])
def test_parenthetical_address_reference_and_numbered_details(suffix):
    address = f"서울특별시 강남구 가상누리로 9999{suffix}"
    text = f"주소: {address} 확인"
    assert {text[r.start:r.end] for r in recognize_addresses(text)} == {address}


@pytest.mark.parametrize("comment", [
    "(문앞에 놓아주세요)", "(문앞에)", "(배송 완료)", "(참고: 주말 배송)",
    "(연락처 010-1234-5678)", "(가상동, 문앞에 놓아주세요)",
])
def test_generic_parenthetical_comments_are_not_absorbed(comment):
    address = "서울 강남구 가상누리로 9999"
    text = f"{address} {comment}"
    assert [text[r.start:r.end] for r in recognize_addresses(text)] == [address]


@pytest.mark.parametrize("address", [
    "서울특별시 강남구\n가상누리로 9999 101동 202호",
    "서울특별시\n강남구\n가상누리로 9999\n101동 202호",
    "서울특별시 강남구\r\n가상누리로 9999",
    "서울특별시 강남구 가상누리로\n9999",
])
def test_bounded_line_wrapping_preserves_original_offsets(address):
    text = f"📦 배송지: {address}입니다."
    assert {text[r.start:r.end] for r in recognize_addresses(text)} == {address}


@pytest.mark.parametrize("gap", ["\n\n", "\n메모입니다\n", "\n" * 10_000])
def test_address_cannot_jump_paragraphs_or_unrelated_prose(gap):
    assert recognize_addresses(f"서울 강남구{gap}가상누리로 9999") == []


@pytest.mark.parametrize("case_id", ["prose-address-parentheses", "prose-address-line-wrap"])
def test_free_prose_address_tail_is_fully_masked(case_id):
    from ko_pii_guard import KoreanPIIGuard

    path = Path(__file__).resolve().parents[1] / "benchmarks/data/name_address.jsonl"
    row = next(case for line in path.read_text().splitlines()
               if (case := json.loads(line))["id"] == case_id)
    guard = KoreanPIIGuard(entities=["KR_ADDRESS"])
    expected = row["expected"][0]
    found = guard.analyze(row["text"])
    assert [(f.start, f.end) for f in found] == [(expected["start"], expected["end"])]
    masked = guard.mask(row["text"], style="stars")
    assert masked[expected["start"]:expected["end"]] == (
        "*" * (expected["end"] - expected["start"])
    )
