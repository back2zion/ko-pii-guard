"""A learned rejection stage can remove candidates but cannot rewrite accepted spans."""

import math

import pytest
from presidio_analyzer import RecognizerResult


def test_context_filter_rejects_a_title_and_preserves_the_author_and_address():
    torch = pytest.importorskip("torch")
    from ko_pii_guard.name_context import filter_name_results

    class Filter:
        threshold = .99

        def __call__(self, inputs):
            return torch.tensor([math.log(999), -math.log(999)])

    title = RecognizerResult("KR_NAME", 0, 2, .999)
    author = RecognizerResult("KR_NAME", 5, 8, .991)
    address = RecognizerResult("KR_ADDRESS", 10, 12, .98)
    results = filter_name_results([title, author, address], torch.ones(12, 768), Filter())
    assert results == [author, address]
    assert results[0] is author and results[1] is address
    assert (author.start, author.end, author.score) == (5, 8, .991)


def test_missing_complete_context_preserves_every_previous_prediction():
    pytest.importorskip("torch")
    from ko_pii_guard.name_context import filter_name_results

    class Filter:
        def __call__(self, inputs):
            raise AssertionError("partial windows must never reject an existing name")

    results = [RecognizerResult("KR_NAME", 5000, 5002, .99)]
    assert filter_name_results(results, None, Filter()) == results


def test_local_context_keeps_before_and_after_distinct_from_candidate_vectors():
    torch = pytest.importorskip("torch")
    from ko_pii_guard.name_context import span_context_features

    features = torch.arange(12).reshape(6, 2)
    row = span_context_features(features, [(2, 4)], local_context=True)[0]
    assert row[:10].tolist() == [5, 6, 5, 6, 6, 7, 1, 2, 9, 10]
    assert row[10:].tolist() == pytest.approx([1/3, 2/3])
