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
                 device: str = "cpu", batch_size: int = 4, stride: int = 64,
                 name_context_head: Any = None):
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
        self.name_context_head = name_context_head

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
                        local_files_only: bool = True,
                        name_context_path: str | Path | None = None, **kwargs: Any) -> KoreanNER:
        """Load safetensors with no remote code; offline unless explicitly allowed.

        name_context_path explicitly selects an experimental trained character
        name head. It replaces name predictions, preserving address predictions.
        It requires the matching pinned base model, not a custom model_path.
        """
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
        if name_context_path is not None:
            if model_path is not None:
                raise ValueError("Name context checkpoints require the pinned base model")
            from ko_pii_guard.name_context import load_name_head

            kwargs["name_context_head"] = load_name_head(
                name_context_path, model_id=model_id, revision=revision,
                device=kwargs.get("device", "cpu"),
            )
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
        characters: dict[tuple[int, int], tuple[int, str, float]] = {}
        rescue_characters = []
        filter_features, filter_origin = None, 0
        with torch.inference_mode():
            for base in range(0, len(offsets), self.batch_size):
                inputs = {key: value[base:base + self.batch_size].to(self.device)
                          for key, value in encoded.items()}
                if self.name_context_head is None:
                    output = self.model(**inputs)
                else:
                    output = self.model(**inputs, output_hidden_states=True)
                scores, labels = output.logits.softmax(-1).max(-1)
                for row, (row_scores, row_labels) in enumerate(
                    zip(scores.cpu().tolist(), labels.cpu().tolist(), strict=True)
                ):
                    spans = offsets[base + row]
                    valid = [i for i, (start, end) in enumerate(spans) if end > start]
                    if not valid:
                        continue
                    if self.name_context_head is not None:
                        from ko_pii_guard.name_context import HEAD_LABELS, character_features

                        left = min(spans[i][0] for i in valid)
                        right = max(spans[i][1] for i in valid)
                        features, char_ids, positions = character_features(
                            text, output.hidden_states[-1][row], spans, left, right,
                        )
                        # Filtering requires complete context. Partial windows
                        # retain every baseline prediction in long documents.
                        if len(offsets) == 1:
                            filter_features, filter_origin = features, left
                        char_logits = self.name_context_head(
                            features.unsqueeze(0), char_ids.unsqueeze(0), positions.unsqueeze(0),
                            torch.tensor([right-left]),
                        )
                        char_scores, char_labels = char_logits[0].softmax(-1).max(-1)
                        from ko_pii_guard.name_context import rescue_decoders

                        for rescue, threshold in (
                            rescue_decoders(self.name_context_head) if len(offsets) == 1 else ()
                        ):
                            rescue_logits = rescue(
                                features.unsqueeze(0), char_ids.unsqueeze(0),
                                positions.unsqueeze(0),
                                torch.tensor([right-left]),
                            )
                            rescue_scores, rescue_labels = rescue_logits[0].softmax(-1).max(-1)
                            rescue_tokens = {}
                            for i, (score, label) in enumerate(zip(
                                rescue_scores.cpu().tolist(), rescue_labels.cpu().tolist(),
                                strict=True,
                            )):
                                rescue_tokens[left+i, left+i+1] = (0, HEAD_LABELS[label], score)
                            rescue_characters.append((threshold, rescue_tokens))
                        for i, (score, label) in enumerate(zip(
                            char_scores.cpu().tolist(), char_labels.cpu().tolist(), strict=True,
                        )):
                            start = left+i
                            rank = min(i, right-left-1-i)
                            if (start, start+1) not in characters or rank > characters[
                                start, start+1
                            ][0]:
                                characters[start, start+1] = (rank, HEAD_LABELS[label], score)
                    for i in valid:
                        start, end = spans[i]
                        rank = min(i - valid[0], valid[-1] - i)
                        if (start, end) not in tokens or rank > tokens[start, end][0]:
                            tokens[start, end] = (
                                rank, self.model.config.id2label[row_labels[i]], row_scores[i]
                            )
        results = _decode_tokens(tokens, self.score_threshold)
        if self.name_context_head is not None:
            results = [r for r in results if r.entity_type != "KR_NAME"]
            results.extend(_decode_tokens(characters, self.score_threshold))
            results.sort(key=lambda r: r.start)
            from ko_pii_guard.name_context import filter_name_results

            with torch.inference_mode():
                results = filter_name_results(
                    results, filter_features, getattr(self.name_context_head, "span_filter", None),
                    filter_origin,
                )
                for threshold, rescue_tokens in rescue_characters:
                    from ko_pii_guard.name_context import add_nonoverlapping_name_results

                    candidates = _decode_tokens(rescue_tokens, threshold)
                    candidates = filter_name_results(
                        candidates, filter_features,
                        getattr(self.name_context_head, "span_filter", None), filter_origin,
                    )
                    results = add_nonoverlapping_name_results(results, candidates)
        return results


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
