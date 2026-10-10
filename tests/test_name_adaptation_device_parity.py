"""Device diagnostics stay on old development and inspect actual mask differences."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import check_name_adaptation_device_parity
    from check_name_adaptation_device_parity import compare_outputs, sample_ids
finally:
    sys.path[:] = previous_path


def test_sample_is_bounded_and_selected_from_existing_ids_without_labels():
    identifiers = [f"old-{index:04d}" for index in range(1000)]
    development = dict(scope="development", source_changed_during_run=False,
                       selected_ids=identifiers[::-1])
    assert sample_ids(development, 100) == identifiers[:100]
    with pytest.raises(ValueError, match="100"):
        sample_ids(development, 101)
    with pytest.raises(ValueError, match="development"):
        sample_ids({**development, "scope": "heldout"}, 100)
    with pytest.raises(ValueError, match="development"):
        sample_ids({**development, "source_changed_during_run": True}, 100)


def test_actual_mask_disagreement_is_detected_even_when_analyze_spans_match():
    cases = [dict(id="old-1", text="김가람?", expected=[
        dict(entity="KR_NAME", start=0, end=3),
    ])]
    cpu = dict(predictions=[{(0, 3)}], scores=[{(0, 3): 0.999}], masks=["***?"])
    cuda = dict(predictions=[{(0, 3)}], scores=[{(0, 3): 0.999}], masks=["**람?"])
    result = compare_outputs(cases, cpu, cuda)
    assert result["counts"] == dict(span_sentences=0, score_sentences=0,
                                    mask_sentences=1, mask_characters=1)
    assert result["disagreements"][0]["differing_mask_positions"] == [2]
    assert not result["cpu_to_cuda_regression_gate"]["passed"]
    assert result["cuda_to_cpu_regression_gate"]["passed"]


def test_confidence_only_difference_does_not_claim_changed_mask_or_spans():
    cases = [dict(id="old-1", text="홍", expected=[dict(entity="KR_NAME", start=0, end=1)])]
    cpu = dict(predictions=[{(0, 1)}], scores=[{(0, 1): 0.8}], masks=["*"])
    cuda = dict(predictions=[{(0, 1)}], scores=[{(0, 1): 0.801}], masks=["*"])
    result = compare_outputs(cases, cpu, cuda)
    assert result["counts"] == dict(span_sentences=0, score_sentences=1,
                                    mask_sentences=0, mask_characters=0)
    assert result["cpu_to_cuda_regression_gate"]["passed"]
    assert result["cuda_to_cpu_regression_gate"]["passed"]


def test_device_run_uses_explicit_factory_without_changing_candidate_settings(monkeypatch):
    received = []

    def frozen_factory(config):
        received.append(config)
        return SimpleNamespace(decoder_name="coverage", threshold=config["threshold"],
                               encoder_autocast="float32")

    class Guard:
        def __init__(self, **kwargs):
            pass

        def analyze(self, text):
            return []

        def mask(self, text, *, style):
            assert style == "stars"
            return text

    monkeypatch.setattr(check_name_adaptation_device_parity, "KoreanPIIGuard", Guard)
    config = dict(checkpoint="frozen", decoder="coverage", threshold=0.1)
    result = check_name_adaptation_device_parity.run_device(
        [dict(id="old-1", text="안내", expected=[])], config, "cpu", factory=frozen_factory)
    assert received == [{**config, "device": "cpu"}]
    assert config == dict(checkpoint="frozen", decoder="coverage", threshold=0.1)
    assert result["execution"]["adapted_encoder_precision"] == "float32"
