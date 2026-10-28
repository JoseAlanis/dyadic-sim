"""Predict one MADRS item score from patient dialogue text."""

from __future__ import annotations

import argparse
import json

import torch

from dyadic_sim.experiments.dialogue_madrs.data import format_symptom_input
from dyadic_sim.experiments.dialogue_madrs.modeling import load_regression_model
from dyadic_sim.scoring.translator import resolve_device


def predict(
    model_path: str,
    symptom: str,
    patient_dialogue: str,
    device_name: str = "auto",
) -> dict:
    device = resolve_device(device_name)
    tokenizer, model = load_regression_model(model_path)
    model.to(device).eval()
    text = format_symptom_input(
        symptom, patient_dialogue, tokenizer.sep_token or "[SEP]"
    )
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        raw = float(model(**encoded).logits.squeeze().cpu())
    score = max(0.0, min(6.0, raw))
    return {"symptom": symptom, "severity_score": score, "rounded_score": round(score)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--symptom", required=True)
    parser.add_argument("--dialogue", required=True, help="Patient utterances only")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(
        json.dumps(
            predict(args.model, args.symptom, args.dialogue, args.device), indent=2
        )
    )


if __name__ == "__main__":
    main()
