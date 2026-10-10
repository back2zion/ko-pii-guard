"""Dependency-free LoRA for the final XLM-RoBERTa attention query/value layers."""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, *, rank=8, alpha=16):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Expected a linear layer and positive rank/alpha")
        self.base, self.scale, self.enabled = base, alpha / rank, True
        self.base.requires_grad_(False)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs):
        output = self.base(inputs)
        if self.enabled:
            output = output + F.linear(F.linear(inputs, self.lora_A), self.lora_B) * self.scale
        return output


def inject_lora(model, *, rank=8, alpha=16, last_layers=4):
    layers = model.roberta.encoder.layer
    if not 1 <= last_layers <= len(layers):
        raise ValueError("Invalid number of final encoder layers")
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("LoRA is already installed")
    model.requires_grad_(False)
    paths = []
    for index in range(len(layers) - last_layers, len(layers)):
        attention = layers[index].attention.self
        for name in ("query", "value"):
            setattr(attention, name, LoRALinear(getattr(attention, name), rank=rank, alpha=alpha))
            paths.append(f"roberta.encoder.layer.{index}.attention.self.{name}")
    return paths


def lora_state_dict(model):
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
        if name.endswith((".lora_A", ".lora_B"))
    }


def load_lora_state_dict(model, state):
    expected = lora_state_dict(model)
    if not expected or set(expected) != set(state):
        raise ValueError("Adapter checkpoint keys differ from installed LoRA")
    if any(
        not value.is_floating_point()
        or expected[name].shape != value.shape
        or not torch.isfinite(value).all()
        for name, value in state.items()
    ):
        raise ValueError("Invalid adapter tensor shape or nonfinite values")
    model.load_state_dict(state, strict=False)


@contextmanager
def lora_disabled(model):
    modules = [module for module in model.modules() if isinstance(module, LoRALinear)]
    previous = [module.enabled for module in modules]
    try:
        for module in modules:
            module.enabled = False
        yield
    finally:
        for module, enabled in zip(modules, previous, strict=True):
            module.enabled = enabled
