"""Reproduce source-backed, undelimited account/card collision behavior.

This is a synthetic development regression, not an independent production
accuracy estimate. Generated account-shaped numbers are not verified accounts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import synthetic as s  # noqa: E402

from ko_pii_guard import KoreanPIIGuard  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rng = random.Random(20261014)
    guard = KoreanPIIGuard()
    cases = {}
    corpus = []
    for bank in ("농협", "카카오뱅크"):
        hits = covered = json_hits = 0
        card_collisions = 0
        card_context_hits = 0
        for _ in range(300):
            number = s.account(rng, bank, dash=False)
            text = f"{bank} 입금 계좌 {number}"
            start = text.index(number)
            corpus.append(("account", text, number))
            findings = guard.analyze(text)
            hits += any(f.entity == "KR_ACCOUNT" and f.start == start
                        and f.end == start + len(number) for f in findings)
            covered += guard.mask(text, style="stars")[start:] == "*" * len(number)
            json_hits += guard.mask_json({"bank": bank, "account_number": number})[
                "account_number"
            ] == "<KR_ACCOUNT>"
            if any(f.entity == "CREDIT_CARD" and f.text == number for f in findings):
                card_collisions += 1
            card_text = f"{bank} 카드번호 {number}"
            corpus.append(("card", card_text, number))
            card_context_hits += any(f.entity == "CREDIT_CARD" and f.text == number
                                     for f in guard.analyze(card_text))
        cases[bank] = {
            "samples": 300, "account_exact_hits": hits, "account_fully_masked": covered,
            "json_account_hits": json_hits,
            "account_labeled_card_collisions": card_collisions,
            "card_label_card_hits": card_context_hits,
        }
    card_rng = random.Random(20261015)
    card_mislabels = json_card_mislabels = 0
    for _ in range(300):
        number = s.card(card_rng)
        text = f"은행 계좌 확인: {number}"
        corpus.append(("card_16", text, number))
        card_mislabels += any(f.entity == "KR_ACCOUNT" for f in guard.analyze(text))
        json_card_mislabels += guard.mask_json({
            "bank": "카카오뱅크", "account_number": number
        })["account_number"] == "<KR_ACCOUNT>"
    digest = hashlib.sha256(json.dumps(corpus, ensure_ascii=False).encode()).hexdigest()
    report = {
        "scope": "synthetic_development_regression",
        "seed_accounts": 20261014, "seed_cards": 20261015,
        "corpus_sha256": digest, "cases": cases,
        "card_16_samples": 300, "card_16_account_mislabels": card_mislabels,
        "json_card_16_account_mislabels": json_card_mislabels,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.check and (card_mislabels or json_card_mislabels or any(
        c["account_exact_hits"] != c["samples"]
        or c["account_fully_masked"] != c["samples"]
        or c["json_account_hits"] != c["samples"]
        or c["account_labeled_card_collisions"]
        or c["card_label_card_hits"] != 36
        for c in cases.values()
    )):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
