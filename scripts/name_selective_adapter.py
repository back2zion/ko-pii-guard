"""Research selective changes using full-span name and all-O events.

This retains the public baseline anchor path, lock, and temporary LoRA handling
of MultiSourceNER. It does not alter the default runtime or select thresholds.
"""

from __future__ import annotations

from name_adapted_adapter import build_frozen_ner
from name_multisource_adapter import MultiSourceNER, _validate_policy
from name_selective_evidence import all_o_posteriors, select_selective_evidence
from name_span_evidence import span_posteriors
from presidio_analyzer import RecognizerResult


def validate_policy(addition_threshold, removal_threshold, class_logit_correction):
    _validate_policy(addition_threshold, class_logit_correction, True)
    _validate_policy(removal_threshold, class_logit_correction, True)
    if class_logit_correction != 0.:
        raise ValueError("Selective unweighted-head experiment requires zero logit correction")


class SelectiveNER(MultiSourceNER):
    def __init__(self, inner, *, addition_threshold, removal_threshold):
        validate_policy(addition_threshold, removal_threshold, 0.)
        super().__init__(inner, threshold=addition_threshold, class_logit_correction=0.,
                         protect_anchors=True)
        self.addition_threshold = addition_threshold
        self.removal_threshold = removal_threshold
        self.decoder_name = "selective_whole_span"
        self._retained_anchors = ()

    def analyze(self, text):
        with self._inference_lock:
            previous = self._retained_anchors
            self._retained_anchors = ()
            try:
                results = super().analyze(text)
                existing = {(row.start, row.end) for row in results
                            if row.entity_type == "KR_NAME"}
                # Address proposals veto new head proposals, but cannot undo an
                # explicit decision to preserve an already-public baseline name.
                results.extend(RecognizerResult("KR_NAME", start, end, score)
                               for start, end, score in self._retained_anchors
                               if (start, end) not in existing)
                return sorted(results, key=lambda row: (row.start, row.end, row.entity_type))
            finally:
                self._retained_anchors = previous

    def decode_logits(self, logits):
        with self._inference_lock:
            if self.on_logits is not None:
                self.on_logits(logits.detach().cpu().clone())
            negative = all_o_posteriors(logits, [(a, b) for a, b, _ in self._anchors])
            self._retained_anchors = tuple(
                anchor for anchor, probability in zip(self._anchors, negative, strict=True)
                if float(probability) <= self.removal_threshold)
            return select_selective_evidence(span_posteriors(logits), negative,
                                             self.addition_threshold,
                                             self.removal_threshold, self._anchors)


def build_ner(config):
    """Load the frozen encoder only after validating a fully explicit policy."""
    addition = config.get("addition_threshold")
    removal = config.get("removal_threshold")
    validate_policy(addition, removal, config.get("class_logit_correction"))
    if config.get("model_kind") != "frozen":
        raise ValueError("Selective experiment requires explicit frozen model_kind")
    if config.get("algorithm", "selective_whole_span") != "selective_whole_span":
        raise ValueError("Unknown selective policy algorithm")
    options = {key: config[key] for key in ("checkpoint", "path", "device") if key in config}
    options.update(threshold=addition, decoder="viterbi")
    return SelectiveNER(build_frozen_ner(options), addition_threshold=addition,
                        removal_threshold=removal)
