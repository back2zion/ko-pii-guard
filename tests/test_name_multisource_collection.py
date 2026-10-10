"""Consumed data cannot silently become another independent evaluation."""

import pytest

pytest.importorskip("torch")

from scripts.collect_name_multisource_development import validate_consumed  # noqa: E402


def report():
    return dict(scope="second_source_KDPII_frozen_candidate",
                source_changed_during_run=False, candidate={"metrics": {"PS_NAME": {"tp": 1}}},
                selected_ids=["used-1"])


def test_consumed_collection_accepts_exact_completed_ids():
    validate_consumed(report(), [{"id": "used-1", "text": "문장"}])


def test_consumed_collection_rejects_unused_or_reordered_ids():
    with pytest.raises(ValueError, match="previously evaluated"):
        validate_consumed(report(), [{"id": "unused-1", "text": "문장"}])


@pytest.mark.parametrize("change", [
    {"source_changed_during_run": True}, {"candidate": {}}, {"scope": "heldout"},
])
def test_consumed_collection_requires_completed_unchanged_source(change):
    with pytest.raises(ValueError, match="completed frozen"):
        validate_consumed(report() | change, [{"id": "used-1", "text": "문장"}])


def test_consumed_collection_rejects_changed_coordinates():
    with pytest.raises(ValueError, match="identity-normalized"):
        validate_consumed(report(), [{"id": "used-1", "text": "Ａ문장"}])
