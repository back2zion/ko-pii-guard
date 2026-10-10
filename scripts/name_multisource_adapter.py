"""Research whole-span policy using live contextual logits and public E5 anchors.

Anchoring costs three encoder passes per window batch: original public E5,
original prior/address logits, and adapted hidden features. Without anchoring
there are two passes. The adapter never selects policies or promotes a runtime.
"""

from __future__ import annotations

import math

import torch
from name_adapted_adapter import AdaptedNER, build_frozen_ner
from name_adapted_adapter import build_ner as build_adapted_ner
from name_encoder_adaptation import lora_disabled
from name_span_evidence import evidence_spans

from ko_pii_guard import KoreanPIIGuard


def _validate_policy(threshold, class_logit_correction, protect_anchors):
    if (type(threshold) not in (int, float) or not math.isfinite(threshold)
            or not 0 <= threshold <= 1):
        raise ValueError("Explicit finite probability threshold is required")
    if (type(class_logit_correction) not in (int, float)
            or not math.isfinite(class_logit_correction) or class_logit_correction < 0):
        raise ValueError("Explicit finite nonnegative class_logit_correction is required")
    if type(protect_anchors) is not bool:
        raise ValueError("Explicit Boolean protect_anchors is required")


class _OriginalBackboneProxy:
    """Delegate only to the underlying original NER, never the enclosing adapter."""

    def __init__(self, backbone):
        self.backbone = backbone

    def analyze(self, text):
        return self.backbone.analyze(text)


class MultiSourceNER(AdaptedNER):
    """Own one backbone and serialize anchor/context/temporary LoRA state together."""

    def __init__(self, inner, *, threshold, class_logit_correction, protect_anchors):
        _validate_policy(threshold, class_logit_correction, protect_anchors)
        if protect_anchors and inner.backbone.score_threshold != 0.9:
            raise ValueError("Protected public anchors require the fixed raw E5 threshold 0.9")
        super().__init__(inner.backbone, inner.head, inner.vocabulary, threshold=threshold,
                         decoder="viterbi", encoder_autocast=inner.encoder_autocast,
                         on_logits=inner.on_logits)
        # Any owner retaining the original adapter uses the same instance lock.
        self._inference_lock = inner._inference_lock
        self.decoder_name = "whole_span_evidence"
        self.class_logit_correction = class_logit_correction
        self.protect_anchors = protect_anchors
        self._anchors = ()
        self._baseline_guard = KoreanPIIGuard(
            entities=["KR_NAME"], score_threshold=0.,
            ner=_OriginalBackboneProxy(self.backbone),
        ) if protect_anchors else None

    def analyze(self, text):
        with self._inference_lock:
            if not text.strip():
                return []
            anchors = ()
            if self.protect_anchors:
                device_type = torch.device(self.backbone.device).type
                with lora_disabled(self.backbone.model), torch.autocast(
                    device_type=device_type, enabled=False
                ):
                    # Use the same public baseline policy, including name vetoes,
                    # explicit fields, original offsets, and rounded confidence.
                    anchors = tuple((result.start, result.end, result.score)
                                    for result in self._baseline_guard.analyze(text)
                                    if result.entity == "KR_NAME")
            previous = self._anchors
            try:
                self._anchors = anchors
                return super()._analyze_locked(text)
            finally:
                self._anchors = previous

    def decode_logits(self, logits):
        with self._inference_lock:
            if self.on_logits is not None:
                self.on_logits(logits.detach().cpu().clone())
            corrected = logits.clone()
            corrected[:, 1:] -= self.class_logit_correction
            return evidence_spans(corrected, self.threshold, anchors=self._anchors)


def build_ner(config):
    """Build an explicitly selected live policy; no defaults for policy decisions."""
    threshold = config.get("threshold", config.get("cutoff"))
    correction, anchors = config.get("class_logit_correction"), config.get("protect_anchors")
    _validate_policy(threshold, correction, anchors)
    if "threshold" in config and "cutoff" in config and config["threshold"] != config["cutoff"]:
        raise ValueError("Threshold and cutoff aliases must agree")
    kind = config.get("model_kind")
    if kind not in ("adapted", "frozen"):
        raise ValueError("Explicit model_kind must be 'adapted' or 'frozen'")
    options = {key: config[key] for key in ("checkpoint", "path", "device") if key in config}
    options.update(threshold=threshold, decoder="viterbi")
    inner = (build_adapted_ner if kind == "adapted" else build_frozen_ner)(options)
    return MultiSourceNER(inner, threshold=threshold, class_logit_correction=correction,
                          protect_anchors=anchors)
