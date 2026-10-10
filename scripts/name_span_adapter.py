"""Opt-in research adapter exercising joint-span results through public masking.

Never enabled by the package. Encoder windows share one scorer and candidate
selection. Existing address predictions are retained and block overlapping names.
"""

from __future__ import annotations

import math

import torch
from name_span_experiment import SpanCandidate, choose_spans
from presidio_analyzer import RecognizerResult

from ko_pii_guard.name_context import character_features
from ko_pii_guard.ner import _decode_tokens


class JointSpanNER:
    def __init__(self, backbone, head, vocabulary, *, max_span_width, threshold):
        if not 0 < threshold < 1:
            raise ValueError("Threshold must be in (0, 1)")
        if not 1 <= max_span_width <= head.widths.num_embeddings:
            raise ValueError("Unsupported candidate width")
        self.backbone = backbone
        self.head = head.to(backbone.device).eval()
        self.vocabulary = vocabulary
        self.max_span_width = max_span_width
        self.threshold = threshold

    def analyze(self, text):
        if not text.strip():
            return []
        ner = self.backbone
        encoded = ner.tokenizer(
            text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=ner.max_length,
            stride=ner.stride,
            padding=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping").tolist()
        encoded.pop("overflow_to_sample_mapping", None)
        margin = math.log(self.threshold / (1 - self.threshold))
        candidates, tokens = {}, {}
        with torch.inference_mode():
            for base in range(0, len(offsets), ner.batch_size):
                inputs = {
                    key: value[base : base + ner.batch_size].to(ner.device)
                    for key, value in encoded.items()
                }
                output = ner.model(**inputs, output_hidden_states=True)
                hidden = output.hidden_states[-1]
                token_scores, token_labels = output.logits.softmax(-1).max(-1)
                for row, spans in enumerate(offsets[base : base + ner.batch_size]):
                    valid_indices = [i for i, (s, e) in enumerate(spans) if e > s]
                    valid_offsets = [spans[i] for i in valid_indices]
                    for i in valid_indices:
                        coordinate = tuple(spans[i])
                        rank = min(i - valid_indices[0], valid_indices[-1] - i)
                        if coordinate not in tokens or rank > tokens[coordinate][0]:
                            tokens[coordinate] = (
                                rank,
                                ner.model.config.id2label[int(token_labels[row, i])],
                                float(token_scores[row, i]),
                            )
                    if not valid_offsets:
                        continue
                    left = min(s for s, _ in valid_offsets)
                    right = max(e for _, e in valid_offsets)
                    features, _, positions = character_features(
                        text, hidden[row], spans, left, right
                    )
                    chars = torch.tensor(
                        [self.vocabulary.get(c, 1) for c in text[left:right]], device=ner.device
                    )
                    # Training caches frozen E5 in fp16; apply the same quantization here.
                    logits, valid = self.head(
                        features.half()[None],
                        chars[None],
                        positions[None],
                        torch.tensor([right - left]),
                        max_span_width=self.max_span_width,
                    )
                    for start, width in (valid[0] & (logits[0] > margin)).nonzero().cpu().tolist():
                        end = start + width + 1
                        coordinate = left + start, left + end
                        rank = min(start, right - left - end)
                        if coordinate not in candidates or rank > candidates[coordinate][0]:
                            candidates[coordinate] = (
                                rank,
                                SpanCandidate(*coordinate, float(logits[0, start, width])),
                            )
        addresses = [
            r for r in _decode_tokens(tokens, ner.score_threshold) if r.entity_type == "KR_ADDRESS"
        ]
        selected = choose_spans(
            [value[1] for value in candidates.values()],
            margin=margin,
            blocked=[(r.start, r.end) for r in addresses],
        )
        results = [
            *addresses,
            *[
                RecognizerResult("KR_NAME", s.start, s.end, 1 / (1 + math.exp(-s.score)))
                for s in selected
            ],
        ]
        return sorted(results, key=lambda r: (r.start, r.end))
