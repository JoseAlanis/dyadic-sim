"""Compare prior intent, therapist coverage, expression, and rater output."""

from __future__ import annotations

import re

from dyadic_sim.scoring.config import FREQUENCY_VALUES
from dyadic_sim.scoring.instrument import Instrument
from dyadic_sim.scoring.instruments import DEFAULT_INSTRUMENT
from dyadic_sim.scoring.types import TurnPair, TurnTopicResult


def compare_prior_expression_prediction(
    prior_levels: dict[str, str],
    prior_source: str,
    pairs: list[TurnPair],
    turn_results: list[TurnTopicResult],
    phq_results: dict[str, dict],
    instrument: Instrument = DEFAULT_INSTRUMENT,
) -> dict[str, dict]:
    prediction_instrument = f"{instrument.name} proxy"
    comparisons: dict[str, dict] = {}
    for symptom, topic in instrument.prior_topic_map.items():
        frequency = prior_levels.get(symptom)
        frequency_value = FREQUENCY_VALUES.get(str(frequency).casefold()) if frequency else None

        if topic is None:
            comparisons[symptom] = {
                "intended_prior": {
                    "instrument": "PHQ-9-style frequency anchor",
                    "value": frequency,
                    "numeric_frequency": frequency_value,
                    "source": prior_source,
                },
                "transcript_expression": {"expressed": None, "relevant_patient_turns": []},
                "therapist_coverage": {"covered": None, "turns": []},
                "prediction": {
                    "instrument": prediction_instrument,
                    "raw_score": None,
                },
                "classification": "not_mappable",
                "scale_comparison_valid": False,
                "mapping_quality": instrument.prior_mapping_quality[symptom],
            }
            continue

        pattern = re.compile(instrument.relevance_patterns[topic], re.IGNORECASE)
        coverage_turns = sorted(
            {
                pair.turn_index
                for pair in pairs
                if pair.valid and pattern.search(pair.therapist_text)
            }
        )
        expressed_turns = sorted(
            {
                result.turn_index
                for result in turn_results
                if result.topic == topic and result.accepted_for_scoring
            }
        )
        prediction = phq_results[symptom]

        intended_present = frequency_value is not None and frequency_value > 0
        if intended_present and not coverage_turns:
            classification = "coverage_failure"
        elif intended_present and coverage_turns and not expressed_turns:
            classification = "simulation_failure"
        elif expressed_turns and prediction["raw_session_score"] is not None:
            classification = "expressed_and_scored"
        elif frequency_value == 0 and expressed_turns:
            classification = "unexpected_expression"
        else:
            classification = "insufficient_information"

        comparisons[symptom] = {
            "intended_prior": {
                "instrument": "PHQ-9-style frequency anchor",
                "value": frequency,
                "numeric_frequency": frequency_value,
                "source": prior_source,
            },
            "transcript_expression": {
                "expressed": bool(expressed_turns),
                "relevant_patient_turns": expressed_turns,
            },
            "therapist_coverage": {
                "covered": bool(coverage_turns),
                "turns": coverage_turns,
            },
            "prediction": {
                "instrument": prediction_instrument,
                "raw_score": prediction["raw_session_score"],
                "rounded_score": prediction["rounded_session_score"],
            },
            "classification": classification,
            "scale_comparison_valid": False,
            "mapping_quality": instrument.prior_mapping_quality[symptom],
        }
    return comparisons

