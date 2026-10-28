"""Regression metrics for overall and per-item MADRS evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any


def _pearson(targets: Sequence[float], predictions: Sequence[float]) -> float | None:
    if len(targets) < 2:
        return None
    target_mean = sum(targets) / len(targets)
    prediction_mean = sum(predictions) / len(predictions)
    numerator = sum(
        (a - target_mean) * (b - prediction_mean)
        for a, b in zip(targets, predictions, strict=True)
    )
    target_ss = sum((value - target_mean) ** 2 for value in targets)
    prediction_ss = sum((value - prediction_mean) ** 2 for value in predictions)
    denominator = math.sqrt(target_ss * prediction_ss)
    return numerator / denominator if denominator else None


def regression_metrics(
    targets: Sequence[float], predictions: Sequence[float]
) -> dict[str, Any]:
    if len(targets) != len(predictions) or not targets:
        raise ValueError("targets and predictions must have the same non-zero length")
    errors = [
        prediction - target
        for target, prediction in zip(targets, predictions, strict=True)
    ]
    rounded_targets = [round(value) for value in targets]
    rounded_predictions = [round(max(0.0, min(6.0, value))) for value in predictions]
    return {
        "count": len(targets),
        "mae": sum(abs(value) for value in errors) / len(errors),
        "rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "pearson": _pearson(targets, predictions),
        "exact_rounded_accuracy": sum(
            a == b for a, b in zip(rounded_targets, rounded_predictions, strict=True)
        )
        / len(targets),
        "within_1_accuracy": sum(
            abs(a - b) <= 1
            for a, b in zip(rounded_targets, rounded_predictions, strict=True)
        )
        / len(targets),
    }


def evaluation_report(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValueError("No prediction rows to evaluate")
    by_symptom: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_symptom[str(row["symptom"])].append(row)

    def metrics_for(items: list[dict[str, Any]]) -> dict[str, Any]:
        return regression_metrics(
            [float(item["severity_score"]) for item in items],
            [float(item["prediction"]) for item in items],
        )

    return {
        "overall": metrics_for(rows),
        "per_symptom": {
            name: metrics_for(items) for name, items in sorted(by_symptom.items())
        },
    }
