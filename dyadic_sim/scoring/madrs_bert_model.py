"""One-process MADRS-BERT regression wrapper.

This is the rater component: one implementation of the interface the pipeline
expects (`prepare_input` + `predict_batch`). The topic taxonomy it scores
against is supplied by the caller, not defined here.
"""

from __future__ import annotations

import math
import re

import torch

from dyadic_sim.scoring.translator import resolve_device, resolved_revision
from dyadic_sim.scoring.types import ModelPrediction


class MadrsBertRegressor:
    """Load MADRS-BERT once and score topic-dialogue inputs in batches."""

    def __init__(
        self,
        model_id: str = "webesama/MADRS-BERT",
        *,
        device: str = "auto",
        batch_size: int | None = None,
        max_length: int = 512,
        score_range: tuple[float, float] = (0.0, 6.0),
        revision: str | None = None,
        tokenizer=None,
        model=None,
    ):
        self.model_id = model_id
        self.revision = revision
        self.device = resolve_device(device)
        self.batch_size = batch_size or (16 if self.device.type == "cuda" else 4)
        self.max_length = max_length
        self.score_range = score_range

        if tokenizer is None or model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id, revision=revision
            )

        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.model.eval()
        self.resolved_revision = resolved_revision(self.model)

    @staticmethod
    def format_input(
        topic: str,
        therapist_text: str,
        patient_text: str,
        earlier_therapist_text: str | None = None,
        earlier_patient_text: str | None = None,
    ) -> str:
        lines = [f"Topic: {topic}", "Dialogue:"]
        if earlier_therapist_text and earlier_patient_text:
            lines.extend(
                [
                    f"Interviewer: {earlier_therapist_text}",
                    f"Patient: {earlier_patient_text}",
                ]
            )
        lines.extend(
            [
                f"Interviewer: {therapist_text}",
                f"Patient: {patient_text}",
            ]
        )
        return "\n".join(lines)

    def would_truncate(self, text: str) -> bool:
        if hasattr(self.tokenizer, "tokenize"):
            token_count = len(self.tokenizer.tokenize(text))
            if hasattr(self.tokenizer, "num_special_tokens_to_add"):
                token_count += self.tokenizer.num_special_tokens_to_add(pair=False)
            else:
                token_count += 2
        else:
            token_count = len(self.tokenizer.encode(text, add_special_tokens=True))
        return token_count > self.max_length

    def _encode_without_length_warning(self, text: str) -> list[int]:
        try:
            return self.tokenizer.encode(
                text,
                add_special_tokens=False,
                verbose=False,
            )
        except TypeError:
            return self.tokenizer.encode(text, add_special_tokens=False)

    def prepare_input(
        self,
        topic: str,
        therapist_text: str,
        patient_text: str,
        earlier_therapist_text: str | None = None,
        earlier_patient_text: str | None = None,
        evidence_text: str | None = None,
    ) -> tuple[str, bool]:
        """Format input while preserving current patient evidence under 512 tokens."""

        full = self.format_input(
            topic,
            therapist_text,
            patient_text,
            earlier_therapist_text,
            earlier_patient_text,
        )
        if not self.would_truncate(full):
            return full, False

        without_earlier = self.format_input(topic, therapist_text, patient_text)
        if not self.would_truncate(without_earlier):
            return without_earlier, True

        header = f"Topic: {topic}\nDialogue:\nInterviewer: \nPatient: "
        header_tokens = len(self.tokenizer.encode(header, add_special_tokens=True))
        available = max(32, self.max_length - header_tokens)
        patient_budget = min(int(available * 0.68), available)
        therapist_budget = max(8, available - patient_budget)

        compact_source = patient_text
        evidence_fragment = None
        if evidence_text and evidence_text in patient_text:
            evidence_fragment = evidence_text
        elif evidence_text:
            evidence_words = set(re.findall(r"\w+", evidence_text.casefold()))
            candidates = [
                item.strip()
                for item in re.split(r"(?<=[.!?])\s+|\n+", patient_text)
                if item.strip()
            ]
            if evidence_words and candidates:
                evidence_fragment = max(
                    candidates,
                    key=lambda item: len(
                        evidence_words
                        & set(re.findall(r"\w+", item.casefold()))
                    ),
                )

        if evidence_fragment:
            evidence_start = patient_text.index(evidence_fragment)
            context_start = max(0, evidence_start - 800)
            context_end = min(
                len(patient_text),
                evidence_start + len(evidence_fragment) + 800,
            )
            compact_source = patient_text[context_start:context_end]

        patient_tokens = self._encode_without_length_warning(compact_source)
        therapist_tokens = self._encode_without_length_warning(therapist_text)
        compact_patient = self.tokenizer.decode(
            patient_tokens[:patient_budget],
            skip_special_tokens=True,
        )
        # Questions and explicit topic cues tend to occur at the end of therapist turns.
        compact_therapist = self.tokenizer.decode(
            therapist_tokens[-therapist_budget:],
            skip_special_tokens=True,
        )
        return self.format_input(topic, compact_therapist, compact_patient), True

    def predict_batch(self, jobs: list[tuple[str, str]]) -> dict[str, ModelPrediction]:
        predictions: dict[str, ModelPrediction] = {}
        for start in range(0, len(jobs), self.batch_size):
            batch = jobs[start : start + self.batch_size]
            job_ids = [item[0] for item in batch]
            texts = [item[1] for item in batch]
            encoded = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                outputs = self.model(**encoded)

            logits = outputs.logits
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits[:, 0]
            elif logits.ndim != 1:
                raise ValueError(
                    f"Expected one regression logit per input; received shape {tuple(logits.shape)}"
                )

            for job_id, value in zip(job_ids, logits.detach().cpu().tolist(), strict=True):
                raw_output = float(value)
                if not math.isfinite(raw_output):
                    raise ValueError(
                        f"Non-finite output from {self.model_id} for job {job_id}"
                    )
                low, high = self.score_range
                clipped = max(low, min(high, raw_output))
                predictions[job_id] = ModelPrediction(
                    job_id=job_id,
                    raw_model_output=raw_output,
                    raw_score=clipped,
                    rounded_score=round(clipped),
                )
        return predictions
