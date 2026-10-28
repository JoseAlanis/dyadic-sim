"""Evaluate a dialogue MADRS checkpoint overall and by symptom."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dyadic_sim.experiments.dialogue_madrs.data import load_samples
from dyadic_sim.experiments.dialogue_madrs.metrics import evaluation_report
from dyadic_sim.experiments.dialogue_madrs.modeling import (
    TokenizedDialogueDataset,
    load_regression_model,
)
from dyadic_sim.scoring.translator import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    tokenizer, model = load_regression_model(args.model)
    model.to(device).eval()
    samples = load_samples(args.test_data)
    from transformers import DataCollatorWithPadding

    loader = DataLoader(
        TokenizedDialogueDataset(samples, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt"),
    )
    predictions: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            values = model(**batch).logits.squeeze(-1).cpu().tolist()
            predictions.extend([float(values)] if isinstance(values, float) else values)

    rows = []
    for sample, raw_prediction in zip(samples, predictions, strict=True):
        row = asdict(sample)
        row["raw_prediction"] = raw_prediction
        row["prediction"] = max(0.0, min(6.0, raw_prediction))
        row["rounded_prediction"] = round(row["prediction"])
        rows.append(row)
    report = evaluation_report(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
