"""Mask JSON-compatible data without serializing away field context."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from ko_pii_guard.normalization import normalize_text

if TYPE_CHECKING:
    from ko_pii_guard.analyzer import KoreanPIIGuard, MaskStyle

# Explicit schema fields supply the same evidence as a label in prose. Never
# infer a person's name from arbitrary keys or from value length alone.
_FIELD_LABELS = {
    "name": "성명", "full_name": "성명", "customer_name": "성명",
    "fullname": "성명", "customername": "성명",
    "성명": "성명", "이름": "이름", "예금주": "예금주",
    "address": "주소", "shipping_address": "주소", "shippingaddress": "주소", "주소": "주소",
    "rrn": "주민번호", "주민번호": "주민번호", "주민등록번호": "주민등록번호",
    "frn": "외국인등록번호", "외국인등록번호": "외국인등록번호",
    "passport": "여권번호", "passport_number": "여권번호",
    "passportnumber": "여권번호", "여권번호": "여권번호",
    "account": "계좌번호", "account_number": "계좌번호",
    "accountnumber": "계좌번호", "계좌번호": "계좌번호",
    "driver_license": "운전면허번호", "driverlicense": "운전면허번호",
    "운전면허번호": "운전면허번호",
    "phone": "전화번호", "phone_number": "전화번호",
    "phonenumber": "전화번호", "전화번호": "전화번호",
    "mobile": "휴대전화", "mobile_number": "휴대전화",
    "mobilenumber": "휴대전화", "휴대전화": "휴대전화",
    "email": "이메일", "email_address": "이메일",
    "emailaddress": "이메일", "이메일": "이메일",
    "card_number": "카드번호", "credit_card_number": "카드번호",
    "cardnumber": "카드번호", "creditcardnumber": "카드번호", "카드번호": "카드번호",
    "brn": "사업자번호", "business_number": "사업자번호",
    "businessnumber": "사업자번호", "사업자번호": "사업자번호",
    "business_registration_number": "사업자등록번호",
    "businessregistrationnumber": "사업자등록번호", "사업자등록번호": "사업자등록번호",
}
_BANK_FIELDS = frozenset(("bank", "bank_name", "bankname", "은행", "은행명"))
_DOCUMENTED_BANK_VALUES = {
    "농협": "농협", "농협은행": "농협", "NH농협은행": "농협",
    "카카오뱅크": "카카오뱅크",
}


def _documented_bank_field(item: dict) -> str:
    """Use only an unambiguous, explicit sibling bank field as account context."""
    banks = set()
    for key, value in item.items():
        if not isinstance(key, str) or key.strip().casefold() not in _BANK_FIELDS:
            continue
        if not isinstance(value, str) or value.strip() not in _DOCUMENTED_BANK_VALUES:
            return ""
        banks.add(_DOCUMENTED_BANK_VALUES[value.strip()])
    return next(iter(banks)) if len(banks) == 1 else ""


def _explicit_card_candidate(text: str) -> bool:
    """Bound a checksum-independent fallback to a whole, explicitly typed field."""
    folded = normalize_text(text).text
    if not re.fullmatch(r"[0-9]+(?:[ -][0-9]+)*", folded):
        return False
    return 13 <= sum(char.isdigit() for char in folded) <= 19


def _explicit_email_candidate(text: str) -> bool:
    """Accept a whole address in a typed field without depending on a TLD list."""
    local, separator, domain = text.partition("@")
    return bool(
        separator and "@" not in domain
        and any(char.isalnum() for char in local)
        and any(char.isalnum() for char in domain)
        and not any(char.isspace() or ord(char) < 32 for char in text)
    )


def transform_json(value: object, transform: Callable[[str, str], str]) -> object:
    """Visit all string keys/values and integers with bounded recursion.

    ``transform`` receives text and an optional canonical Korean field label.
    Numeric values become strings only when a replacement changes them. Dict
    key collisions are errors; silently overwriting a record is never allowed.
    The input is not mutated, even when traversal fails.
    """
    active: set[int] = set()

    def visit(item: object, label: str = "", depth: int = 0) -> object:
        if depth > 100:
            raise ValueError("JSON nesting exceeds 100 levels")
        if isinstance(item, str):
            return transform(item, label)
        if item is None or type(item) is bool:
            return item
        if type(item) is int:
            original = str(item)
            result = transform(original, label)
            return item if result == original else result
        if type(item) is float and math.isfinite(item):
            if label:
                raise ValueError("PII field must not contain a floating-point value")
            return item
        if not isinstance(item, (dict, list)):
            raise TypeError("value must contain only finite JSON-compatible data")
        if id(item) in active:
            raise ValueError("JSON data must not contain cycles")
        active.add(id(item))
        try:
            if isinstance(item, list):
                return [visit(child, label, depth + 1) for child in item]
            result = {}
            bank = _documented_bank_field(item)
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError("JSON object keys must be strings")
                masked_key = transform(key, "")
                if masked_key in result:
                    raise ValueError("Masking would create duplicate JSON object keys")
                child_label = _FIELD_LABELS.get(key.strip().casefold(), "")
                if child_label == "계좌번호" and bank:
                    child_label = f"{bank} {child_label}"
                result[masked_key] = visit(child, child_label, depth + 1)
            return result
        finally:
            active.remove(id(item))

    return visit(value)


def mask_json(guard: KoreanPIIGuard, value: object, style: MaskStyle = "tag") -> object:
    """Return a masked copy of JSON data, including keys and numeric IDs.

    Common explicit fields (``name``, ``rrn``, ``account_number``, etc.) supply
    Korean label context. Names/addresses still require opt-in entities. Numbers
    become strings when masked. If two keys mask to the same key, raise rather
    than losing a value. There are no caller callbacks or implicit model loads.
    """
    if style not in ("tag", "stars", "partial"):
        raise ValueError(f"Unknown mask style: {style}")

    def mask_value(text: str, label: str) -> str:
        prefix = f"{label}: " if label else ""
        result = guard.mask_with_details(prefix + text, style=style)
        if any(redaction.start < len(prefix) for redaction in result.redactions):
            raise ValueError("Detected span crosses a JSON field boundary")
        if (label == "카드번호" and "CREDIT_CARD" in guard.entities
                and not result.redactions and _explicit_card_candidate(text)):
            if style == "tag":
                return "<CREDIT_CARD>"
            if style == "stars":
                return "*" * len(text)
            return guard._partial_value(text, "CREDIT_CARD")
        if (label == "이메일" and "EMAIL_ADDRESS" in guard.entities
                and not result.redactions and _explicit_email_candidate(text)):
            if style == "tag":
                return "<EMAIL_ADDRESS>"
            if style == "stars":
                return "*" * len(text)
            return guard._partial_value(text, "EMAIL_ADDRESS")
        return result.text[len(prefix):]

    return transform_json(value, mask_value)
