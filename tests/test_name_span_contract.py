"""Public span/masking contract with injected annotations, not model accuracy.

The backend in these tests supplies the boundaries explicitly. Passing them
does not establish that a trained model finds these names or their boundaries.
"""

import pytest
from presidio_analyzer import RecognizerResult

from ko_pii_guard import KoreanPIIGuard


class AnnotatedNameBackend:
    def __init__(self, expected_input, start, end):
        self.expected_input = expected_input
        self.start = start
        self.end = end

    def analyze(self, text):
        assert text == self.expected_input
        return [RecognizerResult("KR_NAME", self.start, self.end, 0.97)]


@pytest.mark.parametrize("name,particle", [
    ("이상은", "의"),
    ("김하나", "가"),
    ("박이", "는"),
    ("오지은", "은"),
    ("김은", "과"),
    ("남궁민수", "와"),
])
@pytest.mark.parametrize("style", ["tag", "stars", "partial"])
def test_name_body_keeps_intrinsic_endings_and_excludes_attached_particle(name, particle, style):
    prefix = "📄 동료 "
    suffix = f"{particle} 의견을 확인했습니다."
    text = prefix + name + suffix
    start, end = len(prefix), len(prefix) + len(name)
    guard = KoreanPIIGuard(
        entities=["KR_NAME"], ner=AnnotatedNameBackend(text, start, end)
    )

    [finding] = guard.analyze(text)
    assert (finding.entity, finding.start, finding.end, finding.text) == (
        "KR_NAME", start, end, name,
    )
    assert finding.text == text[finding.start:finding.end]
    assert guard.contains_pii(text)
    replacement = {
        "tag": "<KR_NAME>",
        "stars": "*" * len(name),
        "partial": name[0] + "*" * (len(name) - 1),
    }[style]
    assert guard.mask(text, style=style) == prefix + replacement + suffix


@pytest.mark.parametrize("original_name,normalized_name,partial", [
    ("Ｍｉ\u200bｎａ Example", "Mina Example", "Ｍ*\u200b** *******"),
    ("Jose\u0301 Test", "Jose\u0301 Test", "J***\u0301 ****"),
    ("Nguyễn An", "Nguyễn An", "N***** **"),
])
@pytest.mark.parametrize("style", ["tag", "stars", "partial"])
def test_name_offsets_and_masking_refer_to_original_unicode_text(
    original_name, normalized_name, partial, style
):
    # Folding/removal before the name also shifts the backend's offsets.
    prefix, normalized_prefix = "📎 번호 １２\u200b３, 동료 ", "📎 번호 123, 동료 "
    suffix = "의 의견을 확인했습니다."
    text = prefix + original_name + suffix
    normalized_text = normalized_prefix + normalized_name + suffix
    start = len(normalized_prefix)
    guard = KoreanPIIGuard(
        entities=["KR_NAME"],
        ner=AnnotatedNameBackend(normalized_text, start, start + len(normalized_name)),
    )

    [finding] = guard.analyze(text)
    assert (finding.start, finding.end, finding.text) == (
        len(prefix), len(prefix) + len(original_name), original_name,
    )
    assert finding.text == text[finding.start:finding.end]
    replacement = {
        "tag": "<KR_NAME>", "stars": "*" * len(original_name), "partial": partial,
    }[style]
    assert guard.mask(text, style=style) == prefix + replacement + suffix


@pytest.mark.parametrize("candidate,gold_name", [
    ("김민수는", "김민수"),
    ("이상은의", "이상은"),
    ("김하나가", "김하나"),
])
def test_guard_does_not_guess_or_trim_a_backend_name_boundary(candidate, gold_name):
    # These deliberately overwide candidates violate the semantic gold span.
    # The API preserves accepted model boundaries; it cannot repair the model
    # by blindly stripping characters that may be intrinsic to another name.
    prefix, suffix = "동료 ", " 의견을 확인했습니다."
    text = prefix + candidate + suffix
    start = len(prefix)
    guard = KoreanPIIGuard(
        entities=["KR_NAME"],
        ner=AnnotatedNameBackend(text, start, start + len(candidate)),
    )

    [finding] = guard.analyze(text)
    assert finding.text == candidate
    assert finding.text != gold_name
    assert guard.mask(text) == prefix + "<KR_NAME>" + suffix
