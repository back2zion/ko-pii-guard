"""Research generalization head sharing one E5 forward for names and addresses."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from name_generalization_model import (
    GeneralNameHead,
    character_prior,
    decode_char_logits,
    encode_character_features,
)
from presidio_analyzer import RecognizerResult
from safetensors.torch import load_file

from ko_pii_guard.ner import MODEL_ID, MODEL_REVISION, KoreanNER, _decode_tokens


class GeneralizationNER:
    """Merge window character logits before one globally legal BIOES decode."""

    def __init__(self, backbone, head, vocabulary, *, threshold):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Threshold must be a probability")
        if getattr(backbone, "name_context_head", None) is not None:
            raise ValueError("Generalization head requires raw E5 backbone")
        self.backbone = backbone
        self.head = head.to(backbone.device).eval()
        self.vocabulary = vocabulary
        self.threshold = threshold

    def analyze(self, text):
        if not text.strip():
            return []
        ner = self.backbone
        encoded = ner.tokenizer(
            text, return_offsets_mapping=True, return_overflowing_tokens=True,
            truncation=True, max_length=ner.max_length, stride=ner.stride,
            padding=True, return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping").tolist()
        encoded.pop("overflow_to_sample_mapping", None)
        tokens = {}
        # Leading/trailing uncovered positions remain O in original coordinates.
        merged = torch.full((len(text), 5), -1e4)
        merged[:, 0] = 0.0
        ranks = [-1] * len(text)
        with torch.inference_mode():
            for base in range(0, len(offsets), ner.batch_size):
                inputs = {key: value[base:base + ner.batch_size].to(ner.device)
                          for key, value in encoded.items()}
                output = ner.model(**inputs, output_hidden_states=True)
                hidden = output.hidden_states[-1]
                probabilities = output.logits.softmax(-1)
                token_scores, token_labels = probabilities.max(-1)
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
                    left = min(spans[i][0] for i in valid)
                    right = max(spans[i][1] for i in valid)
                    features, positions = encode_character_features(
                        text, hidden[row], spans, left, right)
                    prior = character_prior(probabilities[row], spans, left, right,
                                            ner.model.config.id2label)
                    chars = torch.tensor([self.vocabulary.get(char, 1)
                                          for char in text[left:right]], device=ner.device)
                    # Match the fp16 frozen-feature training cache before projection.
                    logits = self.head(
                        features.half()[None], chars[None], positions[None], prior[None],
                        torch.tensor([right - left]),
                    )[0].float().cpu()
                    for local, position in enumerate(range(left, right)):
                        rank = min(local, right - left - local - 1)
                        if rank > ranks[position]:
                            merged[position] = logits[local]
                            ranks[position] = rank
        addresses = [result for result in _decode_tokens(tokens, ner.score_threshold)
                     if result.entity_type == "KR_ADDRESS"]
        names = [RecognizerResult("KR_NAME", start, end, score)
                 for start, end, score in decode_char_logits(merged, self.threshold)
                 if not any(start < address.end and address.start < end for address in addresses)]
        return sorted([*addresses, *names], key=lambda result: (result.start, result.end))


def build_ner(config):
    """Evaluation factory: config -> explicitly selected research NER instance."""
    mode = config.get("mode", "learned")
    if mode == "e5":
        return KoreanNER.from_pretrained(device=config.get("device", "cpu"))
    if mode != "learned":
        raise ValueError("Research adapter mode must be 'learned' or 'e5'")
    location = config.get("checkpoint", config.get("path"))
    if location is None:
        raise ValueError("Research adapter requires a checkpoint directory")
    directory = Path(location)
    checkpoint = json.loads((directory / "config.json").read_text())
    if checkpoint.get("model_id") != MODEL_ID or checkpoint.get("revision") != MODEL_REVISION:
        raise ValueError("Generalization checkpoint requires the matching pinned E5 backbone")
    vocabulary = checkpoint["vocabulary"]
    if (not isinstance(vocabulary, dict)
            or any(not isinstance(char, str) or len(char) != 1 for char in vocabulary)
            or set(vocabulary.values()) != set(range(2, len(vocabulary) + 2))):
        raise ValueError("Checkpoint must contain an explicit contiguous character vocabulary")
    head = GeneralNameHead(char_vocab_size=len(vocabulary) + 2, **{
        key: checkpoint[key] for key in (
            "hidden_size", "projection_size", "char_embedding_size", "lstm_hidden_size")
    })
    head.load_state_dict(load_file(str(directory / "head.safetensors")))
    backbone = KoreanNER.from_pretrained(device=config.get("device", "cpu"))
    return GeneralizationNER(backbone, head, vocabulary,
                             threshold=config.get("threshold", checkpoint["threshold"]))
