"""English-to-German translation with batch reuse and traceability."""

from __future__ import annotations

from typing import Protocol

import torch


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolved_revision(model) -> str | None:
    """The commit hash a checkpoint was actually loaded from, if recorded.

    Reported even when no revision was pinned, so a result always says which
    version of a third-party checkpoint produced it. Copy the value into the
    config to pin it.
    """
    for holder in (getattr(model, "config", None), model):
        commit = getattr(holder, "_commit_hash", None)
        if commit:
            return str(commit)
    return None


class Translator(Protocol):
    model_id: str

    def translate_batch(self, texts: list[str]) -> dict[str, str]: ...


class IdentityTranslator:
    model_id = "identity"
    revision = None
    resolved_revision = None

    def translate_batch(self, texts: list[str]) -> dict[str, str]:
        return {text: text for text in dict.fromkeys(texts)}


class MarianEnglishGermanTranslator:
    """Load a Marian-compatible checkpoint once and translate unique strings."""

    def __init__(
        self,
        model_id: str = "Helsinki-NLP/opus-mt-en-de",
        *,
        device: str = "auto",
        batch_size: int | None = None,
        revision: str | None = None,
        tokenizer=None,
        model=None,
    ):
        self.model_id = model_id
        self.revision = revision
        self.device = resolve_device(device)
        self.batch_size = batch_size or (16 if self.device.type == "cuda" else 4)
        self._cache: dict[str, str] = {}

        if tokenizer is None or model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=revision)

        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.model.eval()
        self.resolved_revision = resolved_revision(self.model)

    def translate_batch(self, texts: list[str]) -> dict[str, str]:
        unique = [text for text in dict.fromkeys(texts) if text]
        pending = [text for text in unique if text not in self._cache]

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                generated = self.model.generate(**encoded, max_length=512)
            translated = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            self._cache.update(zip(batch, translated, strict=True))

        return {text: self._cache[text] for text in unique}
