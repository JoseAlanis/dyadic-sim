"""Shared dataset/model helpers for training, evaluation, and inference."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from dyadic_sim.experiments.dialogue_madrs.data import (
    DialogueSample,
    format_symptom_input,
)


class TokenizedDialogueDataset(Dataset):
    def __init__(self, samples: list[DialogueSample], tokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        text = format_symptom_input(
            sample.symptom,
            sample.patient_text,
            self.tokenizer.sep_token or "[SEP]",
        )
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = float(sample.severity_score)
        return encoded


def load_regression_model(model_path: str):
    """Load the existing MADRS-BERT architecture as a one-logit regressor."""

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        problem_type="regression",
        ignore_mismatched_sizes=True,
    )
    model.config.problem_type = "regression"
    model.config.num_labels = 1
    return tokenizer, model
