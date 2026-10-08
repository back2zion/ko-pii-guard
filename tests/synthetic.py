"""Generators for synthetic (non-real) Korean identifiers with valid checksums."""

from __future__ import annotations

import random

RRN_WEIGHTS = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
BRN_KEYS = [1, 3, 7, 1, 3, 7, 1, 3, 5]
DL_REGIONS = [
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "28",
]


def rrn(rng: random.Random, gender_digit: int = 1, dash: bool = True) -> str:
    """Pre-2020-style RRN with a valid checksum and region code."""
    yy = rng.randint(50, 99) if gender_digit in (1, 2, 5, 6) else rng.randint(0, 19)
    mm = rng.randint(1, 12)
    dd = rng.randint(1, 28)
    region = rng.randint(0, 95)
    serial = rng.randint(0, 999)
    body = f"{yy:02d}{mm:02d}{dd:02d}{gender_digit}{region:02d}{serial:03d}"
    check = (11 - sum(int(d) * w for d, w in zip(body, RRN_WEIGHTS, strict=True)) % 11) % 10
    full = body + str(check)
    return f"{full[:6]}-{full[6:]}" if dash else full


def brn(rng: random.Random, dash: bool = True) -> str:
    digits = [rng.randint(0, 9) for _ in range(9)]
    total = sum(d * k for d, k in zip(digits[:8], BRN_KEYS[:8], strict=True))
    last = digits[8] * 5
    total += last // 10 + last
    check = (10 - total % 10) % 10
    s = "".join(map(str, digits)) + str(check)
    return f"{s[:3]}-{s[3:5]}-{s[5:]}" if dash else s


def mobile(rng: random.Random, dash: bool = True) -> str:
    mid, end = rng.randint(2000, 9999), rng.randint(0, 9999)
    return f"010-{mid:04d}-{end:04d}" if dash else f"010{mid:04d}{end:04d}"


def landline(rng: random.Random) -> str:
    return f"02-{rng.randint(2000, 9999):04d}-{rng.randint(0, 9999):04d}"


def driver_license(rng: random.Random) -> str:
    return (
        f"{rng.choice(DL_REGIONS)}-{rng.randint(0, 99):02d}-"
        f"{rng.randint(0, 999999):06d}-{rng.randint(0, 99):02d}"
    )


def passport(rng: random.Random) -> str:
    return f"M{rng.randint(0, 999):03d}{rng.choice('ABCDEFGHJK')}{rng.randint(0, 9999):04d}"


def email(rng: random.Random) -> str:
    return f"user{rng.randint(1, 9999)}@example.co.kr"


def card(rng: random.Random) -> str:
    """Luhn-valid 16-digit card number starting with 4 (test range)."""
    digits = [4] + [rng.randint(0, 9) for _ in range(14)]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    digits.append((10 - total % 10) % 10)
    s = "".join(map(str, digits))
    return f"{s[:4]}-{s[4:8]}-{s[8:12]}-{s[12:]}"
