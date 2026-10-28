"""Session-level symptom severity scoring for dyadic-sim.

The pipeline is instrument-agnostic; see `dyadic_sim.scoring.instrument`.
"""

from dyadic_sim.scoring.config import ScoringConfig
from dyadic_sim.scoring.instrument import Instrument
from dyadic_sim.scoring.instruments import MADRS, get_instrument

__all__ = [
    "MADRS",
    "Instrument",
    "ScoringConfig",
    "SymptomScoringPipeline",
    "get_instrument",
]


def __getattr__(name: str):
    if name == "SymptomScoringPipeline":
        from dyadic_sim.scoring.pipeline import SymptomScoringPipeline

        return SymptomScoringPipeline
    raise AttributeError(name)
