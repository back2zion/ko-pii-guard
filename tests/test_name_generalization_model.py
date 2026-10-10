"""Structural checks for the real-corpus character decoder and frozen E5 prior."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from name_generalization_model import (
        GeneralNameHead,
        character_prior,
        decode_char_logits,
        encode_character_features,
    )
finally:
    sys.path[:] = previous_path


LABELS = dict(enumerate(("O", "B-private_person", "I-private_person", "E-private_person",
                         "S-private_person", "S-private_address")))


def test_token_bioes_prior_expands_to_character_endpoints_and_keeps_spaces_outside():
    probabilities = torch.eye(6)[[0, 1, 3, 4, 4]]
    offsets = [(0, 0), (0, 2), (2, 4), (5, 6), (7, 9)]
    prior = character_prior(probabilities, offsets, 0, 9, LABELS)
    assert prior.argmax(-1).tolist() == [1, 2, 2, 3, 0, 4, 0, 1, 3]
    assert torch.allclose(prior.sum(-1), torch.ones(9))


def test_prior_keeps_probability_mass_and_collapses_other_entities_into_outside():
    probabilities = torch.tensor([[0.1, 0.2, 0.1, 0.2, 0.3, 0.1]])
    prior = character_prior(probabilities, [(10, 13)], 10, 13, LABELS)
    assert torch.allclose(prior[0], torch.tensor([0.2, 0.5, 0.3, 0.0, 0.0]))
    assert torch.allclose(prior[1], torch.tensor([0.2, 0.0, 0.8, 0.0, 0.0]))
    assert torch.allclose(prior[2], torch.tensor([0.2, 0.0, 0.3, 0.5, 0.0]))
    single = character_prior(probabilities, [(10, 11)], 10, 11, LABELS)
    assert torch.allclose(single, torch.tensor([[0.2, 0.2, 0.1, 0.2, 0.3]]))


def test_clipped_window_preserves_original_token_boundaries():
    prior = character_prior(torch.eye(6)[[4]], [(4, 8)], 5, 7, LABELS)
    assert prior.argmax(-1).tolist() == [2, 2]
    features, positions = encode_character_features(
        "가나다라마바사아", torch.tensor([[2.0, 3.0]]), [(4, 8)], left=5, right=7
    )
    assert features.tolist() == [[2.0, 3.0], [2.0, 3.0]]
    assert positions.tolist() == [[0.25, 0.5], [0.5, 0.25]]


def test_overlapping_tokens_choose_same_context_owner_for_prior_and_features():
    offsets = [(0, 2), (1, 3), (3, 4)]
    probabilities = torch.eye(6)[[0, 4, 0]]
    hidden = torch.tensor([[1.0], [2.0], [3.0]])
    prior = character_prior(probabilities, offsets, 0, 4, LABELS)
    features, positions = encode_character_features("가나다라", hidden, offsets)
    assert prior.argmax(-1).tolist() == [0, 1, 3, 0]
    assert features[:, 0].tolist() == [1.0, 2.0, 2.0, 3.0]
    assert positions[1].tolist() == [0.0, 0.5]


def test_uncovered_whitespace_has_zero_features_and_empty_input_is_supported():
    hidden = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    features, positions = encode_character_features("가 나", hidden, [(0, 1), (2, 3)])
    assert features[1].tolist() == positions[1].tolist() == [0.0, 0.0]
    assert encode_character_features("", hidden[:0], [])[0].shape == (0, 2)
    assert character_prior(torch.empty(0, 6), [], 0, 0, LABELS).shape == (0, 5)


def test_decode_closes_entities_handles_singletons_and_respects_confidence():
    logits = torch.full((6, 5), -8.0)
    logits[range(6), [1, 3, 0, 4, 0, 4]] = 8.0
    spans = decode_char_logits(logits, threshold=0.9, origin=10)
    assert [(s, e) for s, e, _ in spans] == [(10, 12), (13, 14), (15, 16)]
    assert all(0.99 < score <= 1.0 for _, _, score in spans)
    assert decode_char_logits(torch.zeros(1, 5), threshold=0.9) == []
    assert decode_char_logits(torch.empty(0, 5)) == []
    # Illegal orphan I must not become an unchecked singleton.
    assert decode_char_logits(torch.tensor([[0.0, -1.0, 12.0, -1.0, 2.0]]),
                              threshold=0.9) == []


def test_head_preserves_prior_residual_and_supports_half_precision_cache_backprop():
    torch.manual_seed(13)
    head = GeneralNameHead(char_vocab_size=10, hidden_size=8, projection_size=4,
                           char_embedding_size=3, lstm_hidden_size=4).eval()
    features = torch.randn(2, 4, 8).half()
    chars = torch.tensor([[1, 2, 3, 4], [1, 2, 0, 0]])
    positions = torch.zeros(2, 4, 2).half()
    prior = torch.tensor([0.5, 0.2, 0.1, 0.1, 0.1]).expand(2, 4, 5).half()
    lengths = torch.tensor([4, 2])
    logits = head(features, chars, positions, prior, lengths)
    assert logits.shape == (2, 4, 5)
    assert logits.dtype == torch.float32
    assert head.prior_scale.item() == pytest.approx(0.25)
    modified = features.clone()
    modified[1, 2:] = 1000
    assert torch.allclose(head(modified, chars, positions, prior, lengths)[1, :2],
                          logits[1, :2])
    loss = torch.nn.functional.cross_entropy(logits[0], torch.tensor([1, 3, 0, 4]))
    loss.backward()
    assert head.prior_scale.grad.abs().item() > 0
    assert head.char_embedding.weight.grad.abs().sum().item() > 0


@pytest.mark.parametrize("bad", [torch.full((1, 6), float("nan")),
                                 torch.tensor([[1.0, -0.1, 0.1, 0, 0, 0]])])
def test_invalid_probabilities_are_rejected(bad):
    with pytest.raises(ValueError):
        character_prior(bad, [(0, 1)], 0, 1, LABELS)


def test_invalid_windows_offsets_and_thresholds_are_rejected():
    with pytest.raises(ValueError):
        character_prior(torch.eye(6)[[0]], [(2, 1)], 0, 2, LABELS)
    with pytest.raises(ValueError):
        encode_character_features("가", torch.zeros(1, 3), [(0, 2)])
    with pytest.raises(ValueError):
        decode_char_logits(torch.zeros(1, 5), threshold=float("nan"))
