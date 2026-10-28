"""Patient-dialogue MADRS item regression experiment.

This package is intentionally separate from :mod:`dyadic_sim.scoring`, whose
existing turn-level scoring pipeline keeps its original input format.
"""

from dyadic_sim.experiments.dialogue_madrs.data import (
    DialogueSample,
    build_dialogue_samples,
    build_synthetic_workbook_samples,
    format_symptom_input,
    grouped_split,
    load_samples,
    save_samples,
)
from dyadic_sim.experiments.dialogue_madrs.metrics import regression_metrics

__all__ = [
    "DialogueSample",
    "build_dialogue_samples",
    "build_synthetic_workbook_samples",
    "format_symptom_input",
    "grouped_split",
    "load_samples",
    "regression_metrics",
    "save_samples",
]
