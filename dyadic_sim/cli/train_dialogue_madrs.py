"""Fine-tune MADRS-BERT on symptom-aware, patient-only dialogue samples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dyadic_sim.experiments.dialogue_madrs.data import load_samples
from dyadic_sim.experiments.dialogue_madrs.metrics import regression_metrics
from dyadic_sim.experiments.dialogue_madrs.modeling import (
    TokenizedDialogueDataset,
    load_regression_model,
)
from dyadic_sim.scoring.translator import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="webesama/MADRS-BERT")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _predict(
    model, loader: DataLoader, device: torch.device
) -> tuple[list[float], list[float]]:
    model.eval()
    targets: list[float] = []
    predictions: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").to(device).float()
            encoded = {key: value.to(device) for key, value in batch.items()}
            logits = model(**encoded).logits.squeeze(-1)
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.cpu().tolist())
    return targets, predictions


def train(args: argparse.Namespace) -> Path:
    """Train with validation-MAE early stopping and save_pretrained checkpoints."""

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    tokenizer, model = load_regression_model(args.model)
    model.to(device)

    train_samples = load_samples(args.data_dir / "train.jsonl")
    validation_samples = load_samples(args.data_dir / "validation.jsonl")
    from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_loader = DataLoader(
        TokenizedDialogueDataset(train_samples, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        TokenizedDialogueDataset(validation_samples, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_dir = args.output_dir / "best_model"
    history: list[dict] = []
    best_mae = math.inf
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(output.loss.detach().cpu())

        targets, predictions = _predict(model, validation_loader, device)
        metrics = regression_metrics(targets, predictions)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(train_loader)),
            **{f"validation_{key}": value for key, value in metrics.items()},
        }
        history.append(record)
        print(json.dumps(record))
        if metrics["mae"] < best_mae - args.min_delta:
            best_mae = metrics["mae"]
            stale_epochs = 0
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(
                    f"Early stopping after epoch {epoch}; "
                    f"best validation MAE={best_mae:.4f}"
                )
                break

    # Retain a deployable best checkpoint and final optimizer state for provenance.
    last_dir = args.output_dir / "last_model"
    model.save_pretrained(last_dir)
    tokenizer.save_pretrained(last_dir)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
        },
        args.output_dir / "training_state.pt",
    )
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    return best_dir


if __name__ == "__main__":
    train(parse_args())
