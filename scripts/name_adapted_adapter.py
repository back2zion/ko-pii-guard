"""Research LoRA inference with original E5 priors and address predictions.

The adapter owns its backbone exclusively. A per-instance lock serializes the
temporary LoRA disable/enable state across concurrent analyze calls.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import torch
from name_coverage_decoding import coverage_spans
from name_encoder_adaptation import inject_lora, load_lora_state_dict, lora_disabled
from name_generalization_model import (
    GeneralNameHead,
    character_prior,
    decode_char_logits,
    encode_character_features,
)
from presidio_analyzer import RecognizerResult
from safetensors.torch import load_file

from ko_pii_guard.ner import MODEL_ID, MODEL_REVISION, KoreanNER, _decode_tokens


class AdaptedNER:
    """Two encoder passes separate adapted features from original E5 decisions."""

    def __init__(self, backbone, head, vocabulary, *, threshold, decoder="viterbi",
                 encoder_autocast="bfloat16", on_logits=None):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Threshold must be a probability")
        if decoder not in ("viterbi", "coverage"):
            raise ValueError("Unknown name decoder")
        if encoder_autocast not in ("bfloat16", "float32"):
            raise ValueError("Unsupported adapted encoder precision")
        if getattr(backbone, "name_context_head", None) is not None:
            raise ValueError("Adapted head requires the original E5 backbone")
        if on_logits is not None and not callable(on_logits):
            raise ValueError("on_logits must be callable")
        self.backbone = backbone
        self.head = head.to(backbone.device).eval()
        self.vocabulary = vocabulary
        self.threshold = threshold
        self.decoder_name = decoder
        self.encoder_autocast = encoder_autocast
        self.on_logits = on_logits
        self._inference_lock = threading.RLock()

    def decode_logits(self, logits):
        """Decode merged original-coordinate logits; observers cannot mutate them."""
        if self.on_logits is not None:
            self.on_logits(logits.detach().cpu().clone())
        decoder = coverage_spans if self.decoder_name == "coverage" else decode_char_logits
        return decoder(logits, self.threshold)

    def analyze(self, text):
        with self._inference_lock:
            return self._analyze_locked(text)

    def _analyze_locked(self, text):
        if not text.strip():
            return []
        ner = self.backbone
        device_type = torch.device(ner.device).type
        encoded = ner.tokenizer(
            text, return_offsets_mapping=True, return_overflowing_tokens=True,
            truncation=True, max_length=ner.max_length, stride=ner.stride,
            padding=True, return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping").tolist()
        encoded.pop("overflow_to_sample_mapping", None)
        tokens = {}
        merged = torch.full((len(text), 5), -1e4)
        merged[:, 0] = 0.
        ranks = [-1] * len(text)
        with torch.inference_mode():
            for base in range(0, len(offsets), ner.batch_size):
                inputs = {key: value[base:base + ner.batch_size].to(ner.device)
                          for key, value in encoded.items()}
                # Preserve the exact original classifier prior and address logits.
                with lora_disabled(ner.model), torch.autocast(device_type=device_type,
                                                              enabled=False):
                    original = ner.model(**inputs)
                probabilities = original.logits.float().softmax(-1)
                token_scores, token_labels = probabilities.max(-1)
                # Training adapts only hidden features under CUDA bfloat16 autocast.
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                                    enabled=(device_type == "cuda"
                                             and self.encoder_autocast == "bfloat16")):
                    hidden = ner.model.base_model(**inputs).last_hidden_state
                for row, spans in enumerate(offsets[base:base + ner.batch_size]):
                    valid = [i for i, (start, end) in enumerate(spans) if end > start]
                    if not valid:
                        continue
                    for index in valid:
                        coordinate = tuple(spans[index])
                        rank = min(index - valid[0], valid[-1] - index)
                        if coordinate not in tokens or rank > tokens[coordinate][0]:
                            tokens[coordinate] = (
                                rank, ner.model.config.id2label[int(token_labels[row, index])],
                                float(token_scores[row, index]),
                            )
                    # Training sees complete sentences, including uncovered edge
                    # spaces. Match that context when no window clipping is needed.
                    left = 0 if len(offsets) == 1 else min(spans[i][0] for i in valid)
                    right = len(text) if len(offsets) == 1 else max(spans[i][1] for i in valid)
                    features, positions = encode_character_features(
                        text, hidden[row], spans, left, right)
                    prior = character_prior(probabilities[row], spans, left, right,
                                            ner.model.config.id2label)
                    chars = torch.tensor([self.vocabulary.get(char, 1)
                                          for char in text[left:right]], device=ner.device)
                    logits = self.head(features.half()[None], chars[None], positions[None],
                                       prior[None], torch.tensor([right - left]))[0].float().cpu()
                    for local, position in enumerate(range(left, right)):
                        rank = min(local, right - left - local - 1)
                        if rank > ranks[position]:
                            merged[position] = logits[local]
                            ranks[position] = rank
        addresses = [result for result in _decode_tokens(tokens, ner.score_threshold)
                     if result.entity_type == "KR_ADDRESS"]
        names = [RecognizerResult("KR_NAME", start, end, score)
                 for start, end, score in self.decode_logits(merged)
                 if not any(start < address.end and address.start < end for address in addresses)]
        return sorted([*addresses, *names], key=lambda result: (result.start, result.end))


def build_ner(config):
    """Construct only an explicitly selected adapted research checkpoint."""
    location = config.get("checkpoint", config.get("path"))
    if location is None:
        raise ValueError("Adapted NER requires a checkpoint directory")
    directory = Path(location)
    checkpoint = json.loads((directory / "config.json").read_text())
    if checkpoint.get("model_id") != MODEL_ID or checkpoint.get("revision") != MODEL_REVISION:
        raise ValueError("Adapted checkpoint requires the matching pinned E5 backbone")
    if checkpoint.get("original_prior_required") is not True:
        raise ValueError("Adapted checkpoint must retain the original E5 prior")
    vocabulary = checkpoint["vocabulary"]
    if (not isinstance(vocabulary, dict)
            or any(not isinstance(char, str) or len(char) != 1 for char in vocabulary)
            or set(vocabulary.values()) != set(range(2, len(vocabulary) + 2))):
        raise ValueError("Checkpoint requires an explicit contiguous character vocabulary")
    adaptation = checkpoint.get("adaptation", {})
    if not {"rank", "alpha", "last_layers", "modules"} <= adaptation.keys():
        raise ValueError("Checkpoint must identify its adapted encoder layers")
    head = GeneralNameHead(char_vocab_size=len(vocabulary) + 2, **{
        key: checkpoint[key] for key in (
            "hidden_size", "projection_size", "char_embedding_size", "lstm_hidden_size")
    })
    head.load_state_dict(load_file(str(directory / "head.safetensors")))
    backbone = KoreanNER.from_pretrained(device=config.get("device", "cpu"))
    installed = inject_lora(backbone.model, **{
        key: adaptation[key] for key in ("rank", "alpha", "last_layers")
    })
    if installed != adaptation["modules"]:
        raise ValueError("Checkpoint adapter module paths differ from installed LoRA")
    load_lora_state_dict(backbone.model, load_file(str(directory / "lora.safetensors")))
    backbone.model.eval()
    return AdaptedNER(
        backbone, head, vocabulary, threshold=config.get("threshold", checkpoint["threshold"]),
        decoder=config.get("decoder", checkpoint.get("decoder", "viterbi")),
        encoder_autocast=checkpoint.get("encoder_autocast", "float32"),
    )


def build_frozen_ner(config):
    """Load the original frozen-feature head with an explicit posterior cutoff.

    This fallback deliberately uses the same two-pass inference and character
    window logic as the adapted candidate, with no LoRA installed. It therefore
    costs two original E5 encoder passes per window batch.
    """
    if "threshold" not in config:
        raise ValueError("Frozen fallback requires an explicit threshold")
    location = config.get("checkpoint", config.get("path"))
    if location is None:
        raise ValueError("Frozen fallback requires a checkpoint directory")
    directory = Path(location)
    checkpoint = json.loads((directory / "config.json").read_text())
    if checkpoint.get("model_id") != MODEL_ID or checkpoint.get("revision") != MODEL_REVISION:
        raise ValueError("Frozen checkpoint requires the matching pinned E5 backbone")
    if "adaptation" in checkpoint:
        raise ValueError("Frozen fallback cannot load an adapted checkpoint")
    vocabulary = checkpoint["vocabulary"]
    if (not isinstance(vocabulary, dict)
            or any(not isinstance(char, str) or len(char) != 1 for char in vocabulary)
            or set(vocabulary.values()) != set(range(2, len(vocabulary) + 2))):
        raise ValueError("Checkpoint requires an explicit contiguous character vocabulary")
    head = GeneralNameHead(char_vocab_size=len(vocabulary) + 2, **{
        key: checkpoint[key] for key in (
            "hidden_size", "projection_size", "char_embedding_size", "lstm_hidden_size")
    })
    head.load_state_dict(load_file(str(directory / "head.safetensors")))
    backbone = KoreanNER.from_pretrained(device=config.get("device", "cpu"))
    return AdaptedNER(backbone, head, vocabulary, threshold=config["threshold"],
                      decoder=config.get("decoder", "coverage"), encoder_autocast="float32")
