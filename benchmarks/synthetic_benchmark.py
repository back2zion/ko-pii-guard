"""Recall / false-positive check on synthetic Korean sentences.

All identifiers are randomly generated with valid checksums and do not belong
to real people. Run:  python benchmarks/synthetic_benchmark.py
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import synthetic as s  # noqa: E402

from ko_pii_guard import KoreanPIIGuard  # noqa: E402

N = 200  # samples per case

POSITIVE_CASES = {
    "KR_RRN": [
        ("주민번호 {} 로 본인확인 부탁드립니다", lambda r: s.rrn(r)),
        ("번호는 {} 입니다", lambda r: s.rrn(r, dash=False)),
    ],
    "KR_FRN": [
        ("외국인등록번호 {} 확인", lambda r: s.rrn(r, gender_digit=5)),
        ("등록번호 {}", lambda r: s.rrn(r, gender_digit=6)),
    ],
    "KR_BRN": [
        ("사업자등록번호 {}", lambda r: s.brn(r)),
        ("거래처 번호 {}", lambda r: s.brn(r, dash=False)),
    ],
    "PHONE_NUMBER": [
        ("연락처 {} 로 전화주세요", lambda r: s.mobile(r)),
        ("내일 {} 로 보내 주세요", lambda r: s.mobile(r, dash=False)),
        ("사무실 {}", lambda r: s.landline(r)),
    ],
    "KR_DRIVER_LICENSE": [
        ("운전면허번호 {}", lambda r: s.driver_license(r)),
        ("번호 {} 확인", lambda r: s.driver_license(r)),
    ],
    "KR_PASSPORT": [("여권번호 {}", lambda r: s.passport(r)), ("코드 {}", lambda r: s.passport(r))],
    "EMAIL_ADDRESS": [("메일 {} 로 회신", lambda r: s.email(r))],
    "CREDIT_CARD": [("카드번호 {} 결제", lambda r: s.card(r))],
}

NEGATIVE_TEMPLATES = [
    lambda r: f"오늘 매출은 {r.randint(1, 9_999_999):,}원입니다",
    lambda r: (
        f"회의는 2026-{r.randint(1, 12):02d}-{r.randint(1, 28):02d} {r.randint(9, 18)}:00에 합니다"
    ),
    lambda r: f"주문번호 2026{r.randint(0, 99999999):08d} 확인 바랍니다",
    lambda r: f"버전 {r.randint(1, 9)}.{r.randint(0, 20)}.{r.randint(0, 99)}로 업데이트했습니다",
    lambda r: f"재고 {r.randint(1, 99999)}개, 단가 {r.randint(100, 99999)}원",
    lambda r: f"송장번호 {r.randint(10**11, 10**12 - 1)}",
]


def main() -> None:
    rng = random.Random(2026)
    guard = KoreanPIIGuard()
    print("| Entity | Case | Detected | Recall |")
    print("|---|---|---|---|")
    for entity, cases in POSITIVE_CASES.items():
        for template, gen in cases:
            hit = 0
            for _ in range(N):
                text = template.format(gen(rng))
                if entity in {f.entity for f in guard.analyze(text)}:
                    hit += 1
            print(f"| {entity} | `{template.format('…')}` | {hit}/{N} | {hit / N:.0%} |")

    print()
    fp = defaultdict(int)
    total = 0
    for make in NEGATIVE_TEMPLATES:
        for _ in range(N):
            text = make(rng)
            total += 1
            for f in guard.analyze(text):
                fp[f.entity] += 1
    flagged = sum(fp.values())
    print(
        f"False positives on {total} PII-free sentences: {flagged} "
        f"({flagged / total:.1%}) {dict(fp) if fp else ''}"
    )


if __name__ == "__main__":
    main()
