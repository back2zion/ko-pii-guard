"""Structured data and CLI contracts for real file-processing use."""

import csv
import io
import json
import random
import subprocess
import sys

import pytest
import synthetic as s
from presidio_analyzer import RecognizerResult

from ko_pii_guard import KoreanPIIGuard, cli
from ko_pii_guard.structured import _FIELD_ENTITIES, _FIELD_LABELS, transform_json


def run_cli(*args, input_text=""):
    return subprocess.run(
        [sys.executable, "-m", "ko_pii_guard", *map(str, args)],
        input=input_text, text=True, capture_output=True, check=False,
    )


def test_json_masks_nested_values_keys_and_numeric_identifiers_without_mutation():
    guard = KoreanPIIGuard(entities=["KR_NAME", "PHONE_NUMBER"])
    source = {"name": "이상", "contacts": ["010-1234-5678"],
              "010-0000-0000": "ok", "count": 2}
    result = guard.mask_json(source)
    assert result == {"name": "<KR_NAME>", "contacts": ["<PHONE_NUMBER>"],
                      "<PHONE_NUMBER>": "ok", "count": 2}
    assert source["name"] == "이상"


def test_generic_json_transform_keeps_stateful_key_callbacks_by_default():
    calls = 0

    def transform(text, label):
        nonlocal calls
        if text == "key" and not label:
            calls += 1
            return f"key-{calls}"
        return text

    source = [{"key": "first"}, {"key": "second"}]
    assert transform_json(source, transform) == [
        {"key-1": "first"}, {"key-2": "second"}
    ]
    assert source == [{"key": "first"}, {"key": "second"}]


def test_repeated_pii_object_keys_are_masked_in_each_record():
    source = {"rows": [{"010-1234-5678": "one"},
                       {"010-1234-5678": "two"}]}
    masked = KoreanPIIGuard(entities=["PHONE_NUMBER"]).mask_json(source)
    assert masked == {"rows": [{"<PHONE_NUMBER>": "one"},
                              {"<PHONE_NUMBER>": "two"}]}
    assert source["rows"][0]["010-1234-5678"] == "one"


def test_json_masks_numeric_ids_only_when_detected():
    guard = KoreanPIIGuard(entities=["KR_ACCOUNT"])
    result = guard.mask_json({"account_number": 1234567890123, "count": 42})
    assert result == {"account_number": "<KR_ACCOUNT>", "count": 42}


def test_json_rejects_floating_point_pii_fields_instead_of_leaking_them():
    guard = KoreanPIIGuard()
    with pytest.raises(ValueError, match="floating-point"):
        guard.mask_json({"account_number": 1234567890123.0})
    with pytest.raises(ValueError, match="floating-point"):
        guard.mask_json({"phoneNumber": [1012345678.0]})
    assert guard.mask_json({"amount": 1234.5}) == {"amount": 1234.5}


def test_json_name_parts_mask_single_character_surname_when_selected():
    guard = KoreanPIIGuard(entities=["KR_NAME"])
    source = {"firstName": "길동", "lastName": "홍", "name": "김", "order": "김"}
    assert guard.mask_json(source) == {
        "firstName": "<KR_NAME>", "lastName": "<KR_NAME>",
        "name": "<KR_NAME>", "order": "김",
    }
    assert guard.mask_json(source, style="partial") == {
        "firstName": "길*", "lastName": "*", "name": "*", "order": "김",
    }
    assert KoreanPIIGuard().mask_json({"name": "김"}) == {"name": "김"}


def test_json_explicit_korean_business_fields_supply_context():
    guard = KoreanPIIGuard(entities=["KR_NAME", "KR_ADDRESS", "PHONE_NUMBER",
                                      "EMAIL_ADDRESS", "KR_ACCOUNT", "CREDIT_CARD"])
    source = {
        "고객명": "홍길동", "수취인명": "김영희", "담당자명": "박준호",
        "예금주명": "이민지", "휴대폰번호": "821012345678",
        "이메일주소": "hello@internal.local", "은행계좌번호": "1234567890123",
        "신용카드번호": "4111111111111112",
        "배송주소": "서울특별시 강남구 테헤란로 123",
        "주문번호": "821012345678",
    }
    assert guard.mask_json(source) == {
        "고객명": "<KR_NAME>", "수취인명": "<KR_NAME>",
        "담당자명": "<KR_NAME>", "예금주명": "<KR_NAME>",
        "휴대폰번호": "<PHONE_NUMBER>", "이메일주소": "<EMAIL_ADDRESS>",
        "은행계좌번호": "<KR_ACCOUNT>", "신용카드번호": "<CREDIT_CARD>",
        "배송주소": "<KR_ADDRESS>", "주문번호": "821012345678",
    }
    assert source["고객명"] == "홍길동"


def test_json_common_snake_case_pii_schema_supplies_context():
    rng = random.Random(42)
    source = {
        "person_name": "홍길동",
        "phone_no": "821012345678",
        "bank_account_no": "1234567890123",
        "card_no": "4111111111111112",
        "passport_no": s.passport(rng),
        "foreigner_reg_no": s.rrn(rng, gender_digit=5),
        "business_reg_no": s.brn(rng),
        "driver_license_no": s.driver_license(rng),
        "order_no": "821012345678",
    }
    guard = KoreanPIIGuard(entities=["KR_NAME", "PHONE_NUMBER", "KR_ACCOUNT",
                                      "CREDIT_CARD", "KR_PASSPORT", "KR_FRN",
                                      "KR_BRN", "KR_DRIVER_LICENSE"])
    assert guard.mask_json(source) == {
        "person_name": "<KR_NAME>", "phone_no": "<PHONE_NUMBER>",
        "bank_account_no": "<KR_ACCOUNT>", "card_no": "<CREDIT_CARD>",
        "passport_no": "<KR_PASSPORT>", "foreigner_reg_no": "<KR_FRN>",
        "business_reg_no": "<KR_BRN>",
        "driver_license_no": "<KR_DRIVER_LICENSE>",
        "order_no": "821012345678",
    }


def test_strict_fields_masks_nonstandard_typed_values_and_respects_entity_selection():
    source = {"person_name": "Alice", "phone_no": ["내선 42", ""],
              "email": "미입력", "order_no": "내선 42", "name": " ",
              "account_number": 12345}
    guard = KoreanPIIGuard(entities=["KR_NAME", "PHONE_NUMBER", "EMAIL_ADDRESS",
                                      "KR_ACCOUNT"])
    assert guard.mask_json(source, strict_fields=True) == {
        "person_name": "<KR_NAME>", "phone_no": ["<PHONE_NUMBER>", ""],
        "email": "<EMAIL_ADDRESS>", "order_no": "내선 42", "name": " ",
        "account_number": "<KR_ACCOUNT>",
    }
    assert guard.mask_json({"phone_no": "???"}, style="partial",
                           strict_fields=True) == {"phone_no": "***"}
    assert KoreanPIIGuard(entities=["PHONE_NUMBER"]).mask_json(
        {"person_name": "Alice", "phone_no": "내선 42"}, strict_fields=True
    ) == {"person_name": "Alice", "phone_no": "<PHONE_NUMBER>"}
    assert source["person_name"] == "Alice"


def test_strict_fields_inherits_explicit_type_into_nested_objects():
    source = {"person_name": {"given": "H0NG", "family": "K1M",
                              "contacts": [{"alias": "김**"}],
                              "phone_no": "내선 42"},
              "order": {"given": "H0NG"}}
    guard = KoreanPIIGuard(entities=["KR_NAME", "PHONE_NUMBER"])
    assert guard.mask_json(source, strict_fields=True) == {
        "person_name": {"given": "<KR_NAME>", "family": "<KR_NAME>",
                        "contacts": [{"alias": "<KR_NAME>"}],
                        "phone_no": "<PHONE_NUMBER>"},
        "order": {"given": "H0NG"},
    }
    assert source["person_name"]["given"] == "H0NG"


def test_every_explicit_field_label_has_a_strict_entity_mapping():
    assert set(_FIELD_LABELS.values()) <= set(_FIELD_ENTITIES)


def test_strict_fields_requires_boolean():
    with pytest.raises(TypeError, match="strict_fields must be a bool"):
        KoreanPIIGuard().mask_json({"phone": "010-1234-5678"}, strict_fields="yes")


@pytest.mark.parametrize("bank,value", [
    ("카카오뱅크", "3333120145448"),
    ("농협", "3025382178204"),
    ("NH농협은행", "3025382178204"),
])
def test_json_uses_explicit_sibling_bank_for_documented_account_collision(bank, value):
    guard = KoreanPIIGuard()
    source = {"bank": bank, "account_number": value}
    assert guard.mask_json(source) == {"bank": bank, "account_number": "<KR_ACCOUNT>"}
    assert source["account_number"] == value


def test_json_bank_mismatch_keeps_card_classification():
    guard = KoreanPIIGuard()
    source = {"bank": "신한", "account_number": "3333120145448"}
    assert guard.mask_json(source)["account_number"] == "<CREDIT_CARD>"


def test_json_bank_context_stays_with_its_own_object():
    guard = KoreanPIIGuard()
    source = [{"bank": "카카오뱅크", "account_number": "3333120145448"},
              {"bank": "신한", "account_number": "3333120145448"}]
    assert [row["account_number"] for row in guard.mask_json(source)] == [
        "<KR_ACCOUNT>", "<CREDIT_CARD>",
    ]


def test_json_camel_case_bank_and_account_fields_preserve_account_type():
    guard = KoreanPIIGuard()
    value = "3333120145448"
    assert guard.mask_json({"bankName": "카카오뱅크", "accountNumber": value}) == {
        "bankName": "카카오뱅크", "accountNumber": "<KR_ACCOUNT>"
    }


def test_json_conflicting_bank_fields_do_not_override_card():
    guard = KoreanPIIGuard()
    source = {"bank": "신한", "bank_name": "카카오뱅크",
              "account_number": "3333120145448"}
    assert guard.mask_json(source)["account_number"] == "<CREDIT_CARD>"


def test_json_phone_field_supplies_context_for_bare_country_code_number():
    guard = KoreanPIIGuard()
    assert guard.mask_json({"phone_number": "821012345678"}) == {
        "phone_number": "<PHONE_NUMBER>"
    }
    assert guard.mask_json({"order_number": "821012345678"}) == {
        "order_number": "821012345678"
    }
    assert guard.mask_json({"phoneNumber": "821012345678"}) == {
        "phoneNumber": "<PHONE_NUMBER>"
    }


@pytest.mark.parametrize("style,expected", [
    ("tag", "<CREDIT_CARD>"),
    ("stars", "****************"),
    ("partial", "41**************"),
])
def test_json_explicit_card_field_masks_checksum_typo(style, expected):
    guard = KoreanPIIGuard()
    invalid_luhn = "4111111111111112"
    assert guard.mask_json({"card_number": invalid_luhn}, style=style) == {
        "card_number": expected
    }
    assert guard.mask_json({"order_number": invalid_luhn}, style=style) == {
        "order_number": invalid_luhn
    }
    if style == "tag":
        assert guard.mask_json({"card_number": int(invalid_luhn)}) == {
            "card_number": expected
        }


def test_json_card_fallback_respects_entity_selection_and_candidate_bounds():
    invalid_luhn = "4111111111111112"
    assert KoreanPIIGuard(entities=["PHONE_NUMBER"]).mask_json({
        "card_number": invalid_luhn
    }) == {"card_number": invalid_luhn}
    assert KoreanPIIGuard().mask_json({"card_number": "123456789012"}) == {
        "card_number": "123456789012"
    }


def test_json_card_fallback_handles_folded_fullwidth_digits():
    value = "４１１１-１１１１-１１１１-１１１２"
    assert KoreanPIIGuard().mask_json({"card_number": value}) == {
        "card_number": "<CREDIT_CARD>"
    }


def test_json_camel_case_card_field_masks_checksum_typo():
    assert KoreanPIIGuard().mask_json({"creditCardNumber": "4111111111111112"}) == {
        "creditCardNumber": "<CREDIT_CARD>"
    }


@pytest.mark.parametrize("style,expected", [
    ("tag", "<EMAIL_ADDRESS>"),
    ("stars", "********************"),
    ("partial", "he***@********.*****"),
])
def test_json_email_field_masks_internal_or_unknown_domains(style, expected):
    value = "hello@internal.local"
    guard = KoreanPIIGuard(entities=["EMAIL_ADDRESS"])
    assert guard.mask_json({"emailAddress": value}, style=style) == {
        "emailAddress": expected
    }
    assert guard.mask_json({"note": value}) == {"note": value}


def test_json_email_fallback_requires_explicit_field_and_selected_entity():
    assert KoreanPIIGuard(entities=["PHONE_NUMBER"]).mask_json({
        "email": "hello@internal.local"
    }) == {"email": "hello@internal.local"}
    assert KoreanPIIGuard(entities=["EMAIL_ADDRESS"]).mask_json({
        "email": "@"
    }) == {"email": "@"}
    assert KoreanPIIGuard(entities=["EMAIL_ADDRESS"]).mask_json({
        "email": "a@b"
    }, style="partial") == {"email": "a@*"}


def test_json_email_field_masks_display_names_and_multiple_internal_addresses():
    guard = KoreanPIIGuard(entities=["EMAIL_ADDRESS"])
    value = "홍길동 <hello@internal.local>; team@office.local"
    assert guard.mask_json({"email": value}) == {"email": "<EMAIL_ADDRESS>"}
    assert guard.mask_json({"email": value}, style="stars") == {
        "email": "*" * len(value)
    }
    assert guard.mask_json({"note": value}) == {"note": value}
    assert guard.mask_json({"email": "홍길동 <hello@example.com>"}) == {
        "email": "<EMAIL_ADDRESS>"
    }


def test_json_key_collision_fails_without_mutating_input():
    source = {"010-1234-5678": 1, "010-5678-1234": 2}
    with pytest.raises(ValueError, match="duplicate JSON object keys"):
        KoreanPIIGuard(entities=["PHONE_NUMBER"]).mask_json(source)
    assert len(source) == 2


def test_json_cycle_and_unsupported_values_fail_explicitly():
    guard = KoreanPIIGuard()
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="cycles"):
        guard.mask_json(cycle)
    with pytest.raises(TypeError, match="JSON-compatible"):
        guard.mask_json({"item": object()})


def test_mask_with_details_has_correct_output_offsets_and_no_raw_values():
    guard = KoreanPIIGuard(entities=["EMAIL_ADDRESS", "PHONE_NUMBER"])
    original = "메일 a@example.com, 연락처 010-1234-5678"
    result = guard.mask_with_details(original)
    assert result.text == "메일 <EMAIL_ADDRESS>, 연락처 <PHONE_NUMBER>"
    assert [result.text[r.output_start:r.output_end] for r in result.redactions] == [
        "<EMAIL_ADDRESS>", "<PHONE_NUMBER>",
    ]
    assert [original[r.start:r.end] for r in result.redactions] == [
        "a@example.com", "010-1234-5678",
    ]
    assert "a@example.com" not in repr(result)
    assert "010-1234-5678" not in repr(result)


def test_mask_many_is_lazy_and_preserves_order():
    guard = KoreanPIIGuard(entities=["PHONE_NUMBER"])
    seen = []

    def documents():
        for document in ("010-1234-5678", "010-5678-1234"):
            seen.append(document)
            yield document

    output = guard.mask_many(documents())
    assert seen == []
    assert next(output) == "<PHONE_NUMBER>"
    assert len(seen) == 1
    assert next(output) == "<PHONE_NUMBER>"


def test_cli_masks_jsonl_to_file_and_keeps_original(tmp_path):
    source = tmp_path / "input.jsonl"
    target = tmp_path / "masked.jsonl"
    source.write_text('{"name":"이상","phone":"010-1234-5678","count":3}\n',
                      encoding="utf-8")
    result = run_cli("mask", source, "--format", "jsonl", "--entity", "KR_NAME",
                     "--entity", "PHONE_NUMBER", "-o", target)
    assert result.returncode == 0, result.stderr
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "name": "<KR_NAME>", "phone": "<PHONE_NUMBER>", "count": 3,
    }
    assert "010-1234-5678" in source.read_text(encoding="utf-8")
    assert target.stat().st_mode & 0o777 == 0o600


def test_cli_masks_multiline_json_with_field_context(tmp_path):
    source = tmp_path / "input.json"
    target = tmp_path / "masked.json"
    source.write_text('''{
  "customer": {"name": "이상", "phoneNumber": "821012345678"},
  "orderNumber": "821012345678"
}
''', encoding="utf-8")
    result = run_cli("mask", source, "--format", "json", "--entity", "KR_NAME",
                     "--entity", "PHONE_NUMBER", "-o", target)
    assert result.returncode == 0, result.stderr
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "customer": {"name": "<KR_NAME>", "phoneNumber": "<PHONE_NUMBER>"},
        "orderNumber": "821012345678",
    }
    assert target.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("format_name,source", [
    ("json", '\ufeff{\n "phone": "010-1234-5678"\n}\n'),
    ("jsonl", '\ufeff{"phone":"010-1234-5678"}\n'),
])
def test_cli_json_formats_accept_utf8_bom_at_file_start(format_name, source):
    result = run_cli("mask", "--format", format_name, input_text=source)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"phone": "<PHONE_NUMBER>"}


def test_cli_jsonl_rejects_bom_inside_later_record_without_stdout():
    source = ('{"phone":"010-1234-5678"}\n'
              '\ufeff{"phone":"010-5678-1234"}\n')
    result = run_cli("mask", "--format", "jsonl", input_text=source)
    assert result.returncode == 2
    assert result.stdout == ""


def test_cli_jsonl_rejects_excessive_nesting_without_traceback_or_partial_output(tmp_path):
    source = ('{"phone":"010-1234-5678"}\n'
              + '[' * 1100 + '"secret"' + ']' * 1100 + '\n')
    result = run_cli("mask", "--format", "jsonl", input_text=source)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
    assert "secret" not in result.stderr
    target = tmp_path / "masked.jsonl"
    target.write_text("previous safe output\n", encoding="utf-8")
    file_result = run_cli("mask", "--format", "jsonl", "-o", target,
                          input_text=source)
    assert file_result.returncode == 2
    assert target.read_text(encoding="utf-8") == "previous safe output\n"
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("contents", [
    '{"phone":"010-1234-5678","phone":"010-5678-1234"}',
    '{"phone":"010-1234-5678","accountNumber":1e999}',
    '{"phone":"010-1234-5678","accountNumber":1234567890123.0}',
    '{"phone":"010-1234-5678"} {"phone":"010-5678-1234"}',
])
def test_cli_invalid_json_preserves_output(contents, tmp_path):
    source = tmp_path / "input.json"
    target = tmp_path / "masked.json"
    source.write_text(contents, encoding="utf-8")
    target.write_text("previous safe output\n", encoding="utf-8")
    result = run_cli("mask", source, "--format", "json", "-o", target)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "010-" not in result.stderr
    assert target.read_text(encoding="utf-8") == "previous safe output\n"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_cli_masks_csv_using_column_context_and_preserves_multiline_fields(tmp_path):
    source = tmp_path / "input.csv"
    target = tmp_path / "masked.csv"
    source.write_text('name,phoneNumber,notes\n'
                      '이상,821012345678,"첫 줄\n둘째 줄"\n', encoding="utf-8")
    result = run_cli("mask", source, "--format", "csv", "--entity", "KR_NAME",
                     "--entity", "PHONE_NUMBER", "-o", target)
    assert result.returncode == 0, result.stderr
    assert target.read_text(encoding="utf-8") == (
        'name,phoneNumber,notes\n<KR_NAME>,<PHONE_NUMBER>,"첫 줄\n둘째 줄"\n'
    )
    assert target.stat().st_mode & 0o777 == 0o600


def test_cli_csv_utf8_bom_does_not_hide_first_column_context():
    result = run_cli("mask", "--format", "csv", "--entity", "KR_NAME",
                     input_text='\ufeffname,notes\n이상,접수\n')
    assert result.returncode == 0, result.stderr
    assert result.stdout == 'name,notes\n<KR_NAME>,접수\n'


def test_cli_csv_korean_headers_mask_typed_values_only():
    result = run_cli("mask", "--format", "csv", "--entity", "KR_NAME",
                     "--entity", "PHONE_NUMBER", "--entity", "EMAIL_ADDRESS",
                     input_text=("고객명,휴대폰번호,이메일주소,주문번호\n"
                                 "홍길동,821012345678,hello@internal.local,821012345678\n"))
    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        "고객명,휴대폰번호,이메일주소,주문번호\n"
        "<KR_NAME>,<PHONE_NUMBER>,<EMAIL_ADDRESS>,821012345678\n"
    )


def test_cli_csv_many_rows_keep_order_and_multiline_text(tmp_path):
    source = tmp_path / "input.csv"
    target = tmp_path / "masked.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["고객명", "휴대폰번호", "주문번호", "notes"])
        for index in range(131):
            writer.writerow(["홍길동", f"010-{index:04d}-1234", f"ORD-{index:04d}",
                             "첫 줄\n둘째 줄" if index == 64 else "접수"])
    result = run_cli("mask", source, "--format", "csv", "--entity", "KR_NAME",
                     "--entity", "PHONE_NUMBER", "-o", target)
    assert result.returncode == 0, result.stderr
    with target.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream, strict=True))
    assert rows[0] == ["고객명", "휴대폰번호", "주문번호", "notes"]
    assert len(rows) == 132
    for index, row in enumerate(rows[1:]):
        assert row == ["<KR_NAME>", "<PHONE_NUMBER>", f"ORD-{index:04d}",
                       "첫 줄\n둘째 줄" if index == 64 else "접수"]


def test_cli_csv_invalid_late_row_preserves_existing_output(tmp_path):
    source = tmp_path / "input.csv"
    target = tmp_path / "masked.csv"
    source.write_text("phone,order\n" +
                      "010-1234-5678,ok\n" * 130 +
                      "010-1234-5678\n", encoding="utf-8")
    target.write_text("previous safe output\n", encoding="utf-8")
    result = run_cli("mask", source, "--format", "csv", "--entity", "PHONE_NUMBER",
                     "-o", target)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "010-" not in result.stderr
    assert target.read_text(encoding="utf-8") == "previous safe output\n"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_cli_csv_accepts_long_text_cell(tmp_path):
    source = tmp_path / "input.csv"
    target = tmp_path / "masked.csv"
    long_note = "x" * 132_000
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["phone", "notes"])
        writer.writerow(["010-1234-5678", long_note])
    result = run_cli("mask", source, "--format", "csv", "--entity", "PHONE_NUMBER",
                     "-o", target)
    assert result.returncode == 0, result.stderr
    previous_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(len(long_note) + 1)
        with target.open(encoding="utf-8", newline="") as stream:
            assert list(csv.reader(stream)) == [["phone", "notes"],
                                                ["<PHONE_NUMBER>", long_note]]
    finally:
        csv.field_size_limit(previous_limit)


@pytest.mark.parametrize("format_name,input_text", [
    ("json", '{"person_name":"Alice","order_no":"Alice"}'),
    ("jsonl", '{"person_name":"Alice","order_no":"Alice"}\n'),
    ("csv", 'person_name,order_no\nAlice,Alice\n'),
])
def test_cli_strict_fields_masks_explicit_values_only(format_name, input_text):
    result = run_cli("mask", "--format", format_name, "--entity", "KR_NAME",
                     "--strict-fields", input_text=input_text)
    assert result.returncode == 0, result.stderr
    if format_name == "csv":
        assert result.stdout == 'person_name,order_no\n<KR_NAME>,Alice\n'
    else:
        assert json.loads(result.stdout) == {
            "person_name": "<KR_NAME>", "order_no": "Alice"
        }


def test_cli_rejects_strict_fields_for_text_mode():
    result = run_cli("mask", "--strict-fields", input_text="Alice")
    assert result.returncode == 2
    assert result.stdout == ""


def test_cli_uses_explicit_offline_ner_configuration(monkeypatch):
    from ko_pii_guard.ner import KoreanNER

    seen = {}

    class FakeNER:
        def analyze(self, text):
            assert text == "홍길동"
            return [RecognizerResult("KR_NAME", 0, 3, 0.95)]

    def load_model(**kwargs):
        seen.update(kwargs)
        return FakeNER()

    output = io.StringIO()
    monkeypatch.setattr(KoreanNER, "from_pretrained", load_model)
    monkeypatch.setattr(sys, "stdin", io.StringIO("홍길동"))
    monkeypatch.setattr(sys, "stdout", output)
    assert cli.main(["mask", "--entity", "KR_NAME", "--ner",
                     "--ner-threshold", "0.8"]) == 0
    assert seen == {"model": "e5", "score_threshold": 0.8}
    assert output.getvalue() == "<KR_NAME>"


@pytest.mark.parametrize("args", [
    ("--ner",),
    ("--ner-model", "kcelectra"),
    ("--ner-threshold", "0.8"),
    ("--entity", "KR_NAME", "--ner", "--ner-threshold", "2"),
])
def test_cli_rejects_inconsistent_ner_options(args):
    assert run_cli("mask", *args, input_text="홍길동").returncode == 2


@pytest.mark.parametrize("contents", [
    'phone,phone\n010-1234-5678,ok\n',
    'phone,notes\n010-1234-5678\n',
    'phone,notes\n010-1234-5678,ok,extra\n',
    'phone,notes\n"010-1234-5678,ok\n',
])
def test_cli_invalid_csv_preserves_output(contents, tmp_path):
    source = tmp_path / "input.csv"
    target = tmp_path / "masked.csv"
    source.write_text(contents, encoding="utf-8")
    target.write_text("previous safe output\n", encoding="utf-8")
    result = run_cli("mask", source, "--format", "csv", "-o", target)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "010-" not in result.stderr
    assert target.read_text(encoding="utf-8") == "previous safe output\n"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_cli_failure_preserves_existing_output_and_emits_no_pii(tmp_path):
    source = tmp_path / "input.jsonl"
    target = tmp_path / "masked.jsonl"
    source.write_text('{"phone":"010-1234-5678"}\n{"secret":"010-5678-1234"\n',
                      encoding="utf-8")
    target.write_text("previous safe output\n", encoding="utf-8")
    result = run_cli("mask", source, "--format", "jsonl", "-o", target)
    assert result.returncode == 2
    assert target.read_text(encoding="utf-8") == "previous safe output\n"
    assert "010-" not in result.stderr
    assert list(tmp_path.glob(".*.tmp")) == []


def test_cli_rejects_broken_output_symlink(tmp_path):
    target = tmp_path / "masked.jsonl"
    target.symlink_to(tmp_path / "missing.jsonl")
    result = run_cli("mask", "--format", "jsonl", "-o", target,
                     input_text='{"phone":"010-1234-5678"}\n')
    assert result.returncode == 2
    assert target.is_symlink()
    assert not (tmp_path / "missing.jsonl").exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_cli_floating_point_pii_failure_preserves_existing_output(tmp_path):
    source = tmp_path / "input.jsonl"
    target = tmp_path / "masked.jsonl"
    source.write_text('{"phone":"010-1234-5678"}\n'
                      '{"accountNumber":1234567890123.0}\n', encoding="utf-8")
    target.write_text("previous safe output\n", encoding="utf-8")
    result = run_cli("mask", source, "--format", "jsonl", "-o", target)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "1234567890123" not in result.stderr
    assert target.read_text(encoding="utf-8") == "previous safe output\n"


def test_cli_stdout_is_empty_if_later_jsonl_record_fails():
    result = run_cli("mask", "--format", "jsonl",
                     input_text='{"phone":"010-1234-5678"}\n{bad json}\n')
    assert result.returncode == 2
    assert result.stdout == ""


def test_cli_analyze_omits_raw_pii_and_check_has_distinct_exit_codes():
    text = "연락처 010-1234-5678"
    analyzed = run_cli("analyze", input_text=text)
    assert analyzed.returncode == 0
    assert "010-1234-5678" not in analyzed.stdout
    assert json.loads(analyzed.stdout)[0]["entity"] == "PHONE_NUMBER"
    assert run_cli("check", input_text=text).returncode == 1
    assert run_cli("check", input_text="개인정보 없음").returncode == 0
