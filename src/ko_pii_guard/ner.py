"""Optional, local context-sensitive Korean person/address recognition.

Model files are fetched only by explicit download(). from_pretrained() reads
local files by default. Inference never sends document text to a service.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from presidio_analyzer import RecognizerResult

MODELS = {
    "e5": ("FrameByFrame/korean-pii-e5-base", "a308c54b4407819624a5661e31e162a269f39818"),
    "kcelectra": ("kiyuyeon/pii-ner-kcelectra", "cf4b4d54fe0c4168c2796a186ea4cb238b6557ca"),
}
MODEL_ID, MODEL_REVISION = MODELS["e5"]
_LABELS = {"NAME": "KR_NAME", "ADDR": "KR_ADDRESS",
           "private_person": "KR_NAME", "private_address": "KR_ADDRESS"}


class KoreanNER:
    """Optional BIO/BIOES classifier with overlapping windows and original spans.

    Use download() once during setup, then from_pretrained() for offline loading.
    Scores are model confidences, not calibrated privacy guarantees.
    """

    def __init__(self, tokenizer: Any, model: Any, *, score_threshold: float = 0.9,
                 device: str = "cpu", batch_size: int = 4, stride: int = 64):
        if not 0 <= score_threshold <= 1:
            raise ValueError("NER score_threshold must be between 0 and 1")
        if not tokenizer.is_fast:
            raise ValueError("NER requires a fast tokenizer with character offsets")
        self.max_length = min(tokenizer.model_max_length, model.config.max_position_embeddings)
        if batch_size < 1 or not 0 < stride < self.max_length - 2:
            raise ValueError("batch_size must be positive; stride must fit the model window")
        self.tokenizer = tokenizer
        self.model = model.to(device).eval()
        self.device = device
        self.score_threshold = score_threshold
        self.batch_size = batch_size
        self.stride = stride

    @staticmethod
    def download(*, model: str = "e5", cache_dir: str | Path | None = None) -> str:
        """Explicitly download a pinned snapshot (default: ~555 MB weights)."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise ImportError("Install the ner extra: uv sync --extra ner") from error
        model_id, revision = _model_source(model)
        return snapshot_download(
            model_id, revision=revision, cache_dir=cache_dir,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "README.md", "LICENSE*"],
        )

    @classmethod
    def from_pretrained(cls, *, model: str = "e5", model_path: str | Path | None = None,
                        local_files_only: bool = True, **kwargs: Any) -> KoreanNER:
        """Load safetensors with no remote code; offline unless explicitly allowed."""
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as error:
            raise ImportError("Install the ner extra: uv sync --extra ner") from error
        model_id, revision = _model_source(model)
        source = str(model_path) if model_path is not None else model_id
        options = {"local_files_only": local_files_only, "trust_remote_code": False}
        if model_path is None:
            options["revision"] = revision
        tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, **options)
        classifier = AutoModelForTokenClassification.from_pretrained(
            source, use_safetensors=True, **options
        )
        labels = set(classifier.config.id2label.values())
        schemas = ({"B-NAME", "I-NAME", "B-ADDR", "I-ADDR"},
                   {"B-private_person", "I-private_person",
                    "B-private_address", "I-private_address"})
        if not any(schema <= labels for schema in schemas):
            raise ValueError("Model must provide supported person/address BIO or BIOES labels")
        return cls(tokenizer, classifier, **kwargs)

    def analyze(self, text: str) -> list[RecognizerResult]:
        if not text.strip():
            return []
        import torch

        encoded = self.tokenizer(
            text, return_offsets_mapping=True, return_overflowing_tokens=True,
            truncation=True, max_length=self.max_length, stride=self.stride,
            padding=True, return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping").tolist()
        encoded.pop("overflow_to_sample_mapping", None)
        # Overlap tokens get the prediction from the window with more context
        # on both sides, rather than whichever window happens to run first.
        tokens: dict[tuple[int, int], tuple[int, str, float]] = {}
        with torch.inference_mode():
            for base in range(0, len(offsets), self.batch_size):
                inputs = {key: value[base:base + self.batch_size].to(self.device)
                          for key, value in encoded.items()}
                scores, labels = self.model(**inputs).logits.softmax(-1).max(-1)
                for row, (row_scores, row_labels) in enumerate(
                    zip(scores.cpu().tolist(), labels.cpu().tolist(), strict=True)
                ):
                    spans = offsets[base + row]
                    valid = [i for i, (start, end) in enumerate(spans) if end > start]
                    if not valid:
                        continue
                    for i in valid:
                        start, end = spans[i]
                        rank = min(i - valid[0], valid[-1] - i)
                        if (start, end) not in tokens or rank > tokens[start, end][0]:
                            tokens[start, end] = (
                                rank, self.model.config.id2label[row_labels[i]], row_scores[i]
                            )
        return _decode_tokens(tokens, self.score_threshold)


def _decode_tokens(tokens: dict[tuple[int, int], tuple[int, str, float]],
                   threshold: float) -> list[RecognizerResult]:
    results = []
    active: str | None = None
    start = end = 0
    scores: list[float] = []

    def flush() -> None:
        if active is not None and scores:
            score = sum(scores) / len(scores)
            if score >= threshold:
                results.append(RecognizerResult(active, start, end, score))

    for (token_start, token_end), (_, label, score) in sorted(tokens.items()):
        prefix, _, name = label.partition("-")
        entity = _LABELS.get(name)
        if entity is None:
            flush()
            active, scores = None, []
            continue
        if prefix in ("B", "S") or entity != active:
            flush()
            active, start, scores = entity, token_start, []
        end = token_end
        scores.append(score)
        if prefix in ("E", "S"):
            flush()
            active, scores = None, []
    flush()
    return results


def _model_source(model: str) -> tuple[str, str]:
    if model not in MODELS:
        raise ValueError(f"Unknown NER model {model!r}; choose from {list(MODELS)}")
    return MODELS[model]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", required=True)
    parser.add_argument("--model", choices=MODELS, default="e5")
    args = parser.parse_args()
    print(KoreanNER.download(model=args.model))


if __name__ == "__main__":
    main()
