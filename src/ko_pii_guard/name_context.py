"""Optional trained character-level name decoder over frozen E5 representations.

Imported only with the ner extra. No name dictionary, word blacklist, network
calls, or tokenizer boundary trimming is used. Checkpoints are explicit opt-in.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

FORMAT_VERSION = 1
CHAR_BUCKETS = 4096
HEAD_LABELS = ("O", "B-private_person", "I-private_person", "E-private_person", "S-private_person")


def character_features(text, hidden, offsets, start=0, end=None):
    """Map token representations to original characters, including subtoken edges.

    For overlapping tokenizer offsets, prefer the narrower token. Whitespace
    omitted by the tokenizer has zero token features and its own char embedding.
    """
    end = len(text) if end is None else end
    length = end - start
    selected = [-1] * length
    widths = [len(text) + 1] * length
    positions = torch.zeros((length, 2), device=hidden.device)
    for token, (left, right) in enumerate(offsets):
        for char in range(max(start, left), min(end, right)):
            i = char - start
            if right - left < widths[i]:
                selected[i], widths[i] = token, right - left
                positions[i, 0] = (char - left) / max(1, right - left)
                positions[i, 1] = (right - char - 1) / max(1, right - left)
    indices = torch.tensor([max(0, i) for i in selected], device=hidden.device)
    features = hidden[indices].clone()
    features[torch.tensor([i < 0 for i in selected], device=hidden.device)] = 0
    chars = torch.tensor(
        [ord(c) % CHAR_BUCKETS for c in text[start:end]], dtype=torch.long, device=hidden.device
    )
    return features, chars, positions


class ContextNameHead(nn.Module):
    """Small bidirectional character decoder; the underlying E5 stays frozen."""

    def __init__(self, hidden_size=768, projection_size=64, char_size=16, recurrent_size=64):
        super().__init__()
        self.projection = nn.Linear(hidden_size, projection_size)
        self.characters = nn.Embedding(CHAR_BUCKETS, char_size)
        self.sequence = nn.LSTM(
            projection_size + char_size + 2, recurrent_size, batch_first=True, bidirectional=True
        )
        self.classifier = nn.Linear(recurrent_size * 2, len(HEAD_LABELS))

    def forward(self, features, characters, positions, lengths):
        inputs = torch.cat(
            (self.projection(features.float()), self.characters(characters), positions.float()),
            dim=-1,
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            inputs, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        decoded, _ = self.sequence(packed)
        decoded, _ = nn.utils.rnn.pad_packed_sequence(
            decoded, batch_first=True, total_length=features.shape[1]
        )
        return self.classifier(decoded)


def span_context_features(features, spans, local_context=False):
    """Use frozen contextual vectors, never a name/word dictionary."""
    features = features.float()
    context = features.mean(dim=0)
    rows = []
    for start, end in spans:
        if not 0 <= start < end <= len(features):
            raise ValueError("Invalid candidate span for context filter")
        local = features[start:end]
        position = features.new_tensor([start / len(features), end / len(features)])
        vectors = [context, local.mean(0), local.amax(0)]
        if local_context:
            before, after = features[max(0, start-16):start], features[end:end+16]
            vectors.extend((before.mean(0) if len(before) else torch.zeros_like(context),
                            after.mean(0) if len(after) else torch.zeros_like(context)))
        rows.append(torch.cat((*vectors, position)))
    return torch.stack(rows)


class ContextSpanFilter(nn.Module):
    """Reject only very confident non-person candidates; retain baseline spans otherwise."""

    def __init__(self, hidden_size=768, intermediate_size=64, threshold=.99, local_context=False):
        super().__init__()
        if not 0 < threshold <= 1:
            raise ValueError("Non-person filter threshold must be in (0, 1]")
        self.threshold = threshold
        self.local_context = local_context
        input_size = hidden_size * (5 if local_context else 3) + 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, intermediate_size), nn.Tanh(),
            nn.Linear(intermediate_size, 1),
        )

    def forward(self, features):
        return self.classifier(features.float()).squeeze(-1)


def filter_name_results(results, features, span_filter, origin=0):
    if features is None or span_filter is None:
        return results
    candidates = [r for r in results if r.entity_type == "KR_NAME"]
    if not candidates:
        return results
    inputs = span_context_features(
        features, [(r.start-origin, r.end-origin) for r in candidates],
        getattr(span_filter, "local_context", False),
    )
    probabilities = span_filter(inputs).sigmoid().cpu().tolist()
    rejected = {id(r) for r, p in zip(candidates, probabilities, strict=True)
                if p >= span_filter.threshold}
    return [r for r in results if id(r) not in rejected]


def load_name_head(path: str | Path, *, model_id: str, revision: str, device: str):
    """Load only tensor weights and a versioned config tied to the base model."""
    path = Path(path)
    config = json.loads((path / "name_context_config.json").read_text(encoding="utf-8"))
    if (
        config.get("format_version") != FORMAT_VERSION
        or config.get("base_model_id") != model_id
        or config.get("base_revision") != revision
        or config.get("labels") != list(HEAD_LABELS)
        or config.get("char_buckets") != CHAR_BUCKETS
    ):
        raise ValueError("Name context checkpoint does not match the pinned base model/schema")
    head = ContextNameHead(**config["architecture"])
    head.load_state_dict(load_file(str(path / "name_context.safetensors")), strict=True)
    if "span_filter" in config:
        span_filter = ContextSpanFilter(**config["span_filter"])
        span_filter.load_state_dict(load_file(str(path / "span_filter.safetensors")), strict=True)
        head.span_filter = span_filter
    if "rescue_head" in config:
        rescue = ContextNameHead(**config["rescue_head"]["architecture"])
        rescue.load_state_dict(
            load_file(str(path / "rescue_name_context.safetensors")), strict=True
        )
        head.rescue_head = rescue
        head.rescue_threshold = config["rescue_head"]["score_threshold"]
    if config.get("extra_rescue_heads"):
        extra_heads, thresholds = [], []
        for index, specification in enumerate(config["extra_rescue_heads"], start=1):
            threshold = specification["score_threshold"]
            if not 0 < threshold <= 1:
                raise ValueError("Rescue threshold must be in (0, 1]")
            extra = ContextNameHead(**specification["architecture"])
            extra.load_state_dict(load_file(str(
                path / f"rescue_name_context_{index}.safetensors"
            )), strict=True)
            extra_heads.append(extra)
            thresholds.append(threshold)
        head.extra_rescue_heads = nn.ModuleList(extra_heads)
        head.extra_rescue_thresholds = thresholds
    return head.to(device).eval()


def rescue_decoders(head):
    """Keep the original rescue first so refinement cannot overwrite its spans."""
    decoders = []
    if getattr(head, "rescue_head", None) is not None:
        decoders.append((head.rescue_head, head.rescue_threshold))
    decoders.extend(zip(getattr(head, "extra_rescue_heads", ()),
                        getattr(head, "extra_rescue_thresholds", ()), strict=True))
    return decoders


def add_nonoverlapping_name_results(existing, candidates):
    """Preserve every accepted name/address; add only disjoint new name spans."""
    accepted = list(existing)
    for candidate in sorted(candidates, key=lambda r: (-r.score, r.start, r.end)):
        if candidate.entity_type != "KR_NAME":
            continue
        if any(candidate.start < old.end and old.start < candidate.end for old in accepted):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda r: r.start)
