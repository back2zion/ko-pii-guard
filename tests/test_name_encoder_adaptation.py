"""LoRA starts identically, updates only adapters, and restores frozen priors."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

previous_path = sys.path[:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from name_encoder_adaptation import (
        LoRALinear,
        inject_lora,
        load_lora_state_dict,
        lora_disabled,
        lora_state_dict,
    )
    from train_name_encoder_adaptation import forward_batch
finally:
    sys.path[:] = previous_path


def test_zero_initialized_adapter_is_identical_and_base_never_gets_gradients():
    base = nn.Linear(5, 3)
    inputs = torch.randn(4, 5)
    expected = base(inputs).detach()
    adapter = LoRALinear(base, rank=2, alpha=4)
    assert torch.equal(adapter(inputs), expected)
    adapter(inputs).sum().backward()
    assert base.weight.grad is base.bias.grad is None
    assert adapter.lora_B.grad.abs().sum() > 0


def test_disabled_context_restores_original_output_even_after_exception():
    model = nn.Sequential(LoRALinear(nn.Linear(3, 2), rank=2, alpha=4))
    inputs = torch.randn(4, 3)
    with torch.no_grad():
        model[0].lora_B.fill_(1.0)
    expected = model[0].base(inputs)
    with pytest.raises(RuntimeError), lora_disabled(model):
        assert torch.equal(model(inputs), expected)
        raise RuntimeError("test")
    assert model[0].enabled
    assert not torch.equal(model(inputs), expected)


def test_adapter_only_checkpoint_round_trip_and_missing_keys_rejected():
    model = nn.Sequential(LoRALinear(nn.Linear(3, 2), rank=2, alpha=4))
    state = lora_state_dict(model)
    assert set(state) == {"0.lora_A", "0.lora_B"}
    copied = {key: value.clone() for key, value in state.items()}
    with torch.no_grad():
        model[0].lora_B.fill_(3.0)
    load_lora_state_dict(model, copied)
    assert torch.equal(model[0].lora_B, copied["0.lora_B"])
    with pytest.raises(ValueError, match="keys"):
        load_lora_state_dict(model, {})


def test_adapter_checkpoint_rejects_nonfloating_weights_without_mutation():
    model = nn.Sequential(LoRALinear(nn.Linear(3, 2), rank=2, alpha=4))
    before = lora_state_dict(model)
    corrupted = {name: value.to(torch.int64) for name, value in before.items()}
    with pytest.raises(ValueError, match="tensor"):
        load_lora_state_dict(model, corrupted)
    assert all(torch.equal(value, lora_state_dict(model)[name]) for name, value in before.items())


def test_injects_only_last_layers_query_value_and_freezes_other_weights():
    model = nn.Module()
    model.roberta = nn.Module()
    model.roberta.encoder = nn.Module()
    layers = []
    for _ in range(3):
        layer = nn.Module()
        layer.attention = nn.Module()
        layer.attention.self = nn.Module()
        for key in ("query", "key", "value"):
            setattr(layer.attention.self, key, nn.Linear(4, 4))
        layers.append(layer)
    model.roberta.encoder.layer = nn.ModuleList(layers)
    injected = inject_lora(model, rank=2, alpha=4, last_layers=2)
    assert len(injected) == 4
    assert isinstance(layers[0].attention.self.query, nn.Linear)
    assert isinstance(layers[-1].attention.self.query, LoRALinear)
    assert isinstance(layers[-1].attention.self.key, nn.Linear)
    assert all("lora_" in name for name, param in model.named_parameters() if param.requires_grad)


def test_padded_encoder_batch_keeps_original_offsets_and_character_gradients():
    from types import SimpleNamespace

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Parameter(torch.randn(2, 4, 3))

        def forward(self, **inputs):
            return SimpleNamespace(last_hidden_state=self.hidden)

    class Tokenizer:
        def pad(self, *args, **kwargs):
            return dict(input_ids=torch.ones(2, 4, dtype=torch.long))

    class Head(nn.Module):
        def forward(self, features, *args):
            return features.float()

    model = SimpleNamespace(roberta=Encoder())
    cases = [dict(text="김"), dict(text="가나")]
    tokens = [
        dict(input_ids=[0, 1, 2], offset_mapping=[(0, 0), (0, 1), (0, 0)]),
        dict(input_ids=[0, 1, 2, 3], offset_mapping=[(0, 0), (0, 1), (1, 2), (0, 0)]),
    ]
    cached = [
        (
            None,
            torch.ones(len(c["text"]), dtype=torch.long),
            None,
            torch.zeros(len(c["text"]), 5),
            torch.zeros(len(c["text"]), dtype=torch.long),
        )
        for c in cases
    ]
    logits, lengths, targets = forward_batch(
        model, Head(), Tokenizer(), tokens, cases, cached, device="cpu"
    )
    assert lengths.tolist() == [1, 2]
    assert targets[0].tolist() == [0, -100]
    logits.sum().backward()
    assert model.roberta.hidden.grad[0, 1].abs().sum() > 0
    assert model.roberta.hidden.grad[0, 3].abs().sum() == 0
