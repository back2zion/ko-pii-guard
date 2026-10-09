"""Conservative Unicode folding with offsets into the untouched input.

Only fullwidth ASCII, decimal digits, common hyphens and non-breaking spaces
are folded. Invisible format characters are removed only inside identifiers.
This deliberately avoids whole-text NFKC, which also changes unrelated prose.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

_DASHES = frozenset("\u2010\u2011\u2012\u2013\u2014\u2212\ufe63")
_SPACES = frozenset("\u00a0\u202f")
_INVISIBLE = frozenset("\u200b\u200c\u200d\u2060\ufeff\u00ad")


def _fold(char: str) -> str:
    code = ord(char)
    if 0xFF01 <= code <= 0xFF5E:
        return chr(code - 0xFEE0)
    if char in _DASHES:
        return "-"
    if char in _SPACES:
        return " "
    if char.isdecimal() and not char.isascii():
        return str(unicodedata.decimal(char))
    return char


def _identifier_char(char: str) -> bool:
    folded = _fold(char)
    return folded.isascii() and (folded.isalnum() or folded in "-@._+")


@dataclass(frozen=True)
class NormalizedText:
    text: str
    positions: tuple[int, ...] | None = None

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        if self.positions is None:
            return start, end
        return self.positions[start], self.positions[end - 1] + 1


def normalize_text(text: str) -> NormalizedText:
    if text.isascii():
        return NormalizedText(text)
    chars: list[str] = []
    positions: list[int] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char in _INVISIBLE:
            end = i + 1
            while end < len(text) and text[end] in _INVISIBLE:
                end += 1
            if i > 0 and end < len(text) and (
                _identifier_char(text[i - 1]) and _identifier_char(text[end])
            ):
                i = end
                continue
            # Preserve the entire run in one pass. Advancing just one codepoint
            # here would rescan every suffix of an unremovable run (quadratic).
            chars.extend(text[i:end])
            positions.extend(range(i, end))
            i = end
            continue
        chars.append(_fold(char))
        positions.append(i)
        i += 1
    folded = "".join(chars)
    return NormalizedText(folded, tuple(positions) if folded != text else None)
