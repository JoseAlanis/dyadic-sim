"""Runtime settings and the PHQ-style prior vocabulary.

This module covers the *prior* side only: the PHQ-9 anchors patient priors are
written in, and the process-wide scoring settings. The rating side (topics,
relevance patterns, score range) lives in `dyadic_sim.scoring.instrument`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dyadic_sim.core.vocabulary import LABELS
from dyadic_sim.scoring.instrument import Instrument
from dyadic_sim.scoring.instruments import DEFAULT_INSTRUMENT

# The PHQ-9 vocabulary lives in dyadic_sim.core.vocabulary, named for its
# role (what priors are written in) rather than for the instrument. It holds all three
# registers (KEYS, LABELS, REFERENCES) side by side so they stop being
# mistaken for copies of one another. Re-exported here under its historical
# name because the prior side has always imported it from config.
PHQ_SYMPTOM_LABELS = LABELS

FREQUENCY_VALUES = {
    "not at all": 0,
    "several days": 1,
    "more than half the days": 2,
    "nearly every day": 3,
}

@dataclass(frozen=True)
class ScoringConfig:
    """Runtime settings shared by every session in one evaluation process."""

    instrument: Instrument = field(default=DEFAULT_INSTRUMENT)
    model_id: str = "webesama/MADRS-BERT"
    translator_model_id: str = "Helsinki-NLP/opus-mt-en-de"
    # Third-party checkpoints can be updated by their owners. Pin a commit SHA
    # here to freeze results; the revision actually used is recorded in every
    # output either way.
    model_revision: str | None = None
    translator_revision: str | None = None
    source_language: str = "en"
    device: str = "auto"
    batch_size: int | None = None
    max_length: int = 512
    relevance_mode: str = "hybrid"
    combined_similarity_threshold: float = 0.50
    patient_similarity_threshold: float = 0.35
    relevance_acceptance_threshold: float = 0.80
    aggregation_method: str = "maximum_relevant_evidence"
