"""Structured data and CLI contracts for real file-processing use."""

import json
import subprocess
import sys

import pytest

from ko_pii_guard import KoreanPIIGuard


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
