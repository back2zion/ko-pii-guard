"""Exercise offset and full-redaction invariants on generated Korean text.

This is a deterministic robustness probe, not an accuracy benchmark. Inputs
are synthetic; no source text or identifier value is emitted in reports.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from ko_pii_guard import KoreanPIIGuard

TOKENS = (
    "연락처 ", "계좌번호: ", "카카오뱅크 입금 계좌 ", "농협 입금 계좌 ",
    "카드번호 ", "주문번호 ", "이메일 ", "메모 ", "\n", ", ", "; ",
    "010-1234-5678", "010 5678 1234", "821012345678", "3333120145448",
    "3025382178204", "4111111111111111", "900101-1234567", "test@example.com",
    "１２３４５６７８９０１２３", "010\u200b-1234-5678", "📎", "일반문장", "값 없음",
)
SEPARATORS = ("-", " ", "\u2010", "\u2011", "\u200b", "\u2060")


def random_token(rng: random.Random) -> str:
    choice = rng.randrange(10)
    if choice < 7:
        return rng.choice(TOKENS)
    if choice < 9:
        digits = "".join(str(rng.randrange(10)) for _ in range(rng.randrange(5, 21)))
        point = rng.randrange(1, len(digits))
        return digits[:point] + rng.choice(SEPARATORS) + digits[point:]
    return "".join(rng.choices("ABCabc가나다123_", k=rng.randrange(2, 15)))


def check_case(guard: KoreanPIIGuard, value: str) -> int:
    findings = guard.analyze(value)
    masked = guard.mask(value, style="stars")
    detailed = guard.mask_with_details(value, style="tag")
    if [(r.entity, r.start, r.end) for r in detailed.redactions] != [
        (f.entity, f.start, f.end) for f in findings
    ]:
        raise AssertionError("Detailed replacements disagree with analyze offsets")
    if len(masked) != len(value):
        raise AssertionError("Stars output changed original character length")
    previous_end = 0
    for finding in findings:
        if not 0 <= finding.start < finding.end <= len(value):
            raise AssertionError("Finding offsets are outside original text")
        if finding.start < previous_end:
            raise AssertionError("Findings overlap or are out of order")
        if finding.text != value[finding.start:finding.end]:
            raise AssertionError("Finding text and offsets disagree")
        if masked[previous_end:finding.start] != value[previous_end:finding.start]:
            raise AssertionError("Non-PII text changed during masking")
        if masked[finding.start:finding.end] != "*" * (finding.end - finding.start):
            raise AssertionError("Detected span was not fully masked")
        previous_end = finding.end
    if masked[previous_end:] != value[previous_end:]:
        raise AssertionError("Trailing non-PII text changed during masking")
    source_cursor = output_cursor = 0
    for redaction in detailed.redactions:
        if (detailed.text[output_cursor:redaction.output_start]
                != value[source_cursor:redaction.start]):
            raise AssertionError("Detailed output changed text outside replacements")
        if detailed.text[redaction.output_start:redaction.output_end] != f"<{redaction.entity}>":
            raise AssertionError("Detailed output replacement has wrong coordinates")
        source_cursor, output_cursor = redaction.end, redaction.output_end
    if detailed.text[output_cursor:] != value[source_cursor:]:
        raise AssertionError("Detailed output changed trailing original text")
    return len(findings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-every", type=int, default=10_000)
    args = parser.parse_args()
    if args.iterations < 1 or args.progress_every < 1:
        parser.error("--iterations and --progress-every must be positive")
    rng = random.Random(args.seed)
    guard = KoreanPIIGuard()
    findings = 0
    for index in range(args.iterations):
        value = "".join(random_token(rng) for _ in range(rng.randrange(1, 18)))
        try:
            findings += check_case(guard, value)
        except Exception as error:
            raise RuntimeError(
                f"Generated masking invariant failed at seed={args.seed}, index={index}: "
                f"{type(error).__name__}"
            ) from None
        if (index + 1) % args.progress_every == 0:
            print(f"checked={index + 1}", file=sys.stderr, flush=True)
    report = {"seed": args.seed, "iterations": args.iterations,
              "findings_checked": findings,
              "scope": "synthetic_analyze_stars_and_tag_detail_invariants"}
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
