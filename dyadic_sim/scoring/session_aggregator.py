"""Maximum-relevant-evidence aggregation and PHQ prior projection."""

from __future__ import annotations

from dyadic_sim.scoring.instrument import Instrument
from dyadic_sim.scoring.instruments import DEFAULT_INSTRUMENT
from dyadic_sim.scoring.types import TopicAggregate, TurnTopicResult


def aggregate_topics(
    results: list[TurnTopicResult],
    instrument: Instrument = DEFAULT_INSTRUMENT,
) -> dict[str, TopicAggregate]:
    aggregates: dict[str, TopicAggregate] = {}
    for topic in instrument.topics:
        eligible = [
            result
            for result in results
            if result.topic == topic
            and result.accepted_for_scoring
            and result.raw_score is not None
        ]
        eligible.sort(
            key=lambda item: (
                -float(item.raw_score),
                -item.relevance_confidence,
                item.turn_index,
            )
        )
        relevant_turns = sorted({item.turn_index for item in eligible})
        if not eligible:
            aggregates[topic] = TopicAggregate(
                topic=topic,
                raw_session_score=None,
                rounded_session_score=None,
                source_turn=None,
                relevant_turns=[],
                evidence=None,
                number_of_relevant_turns=0,
            )
            continue

        maximum = eligible[0]
        aggregates[topic] = TopicAggregate(
            topic=topic,
            raw_session_score=maximum.raw_score,
            rounded_session_score=maximum.rounded_score,
            source_turn=maximum.turn_index,
            relevant_turns=relevant_turns,
            evidence=maximum.evidence,
            number_of_relevant_turns=len(relevant_turns),
            therapist_text=maximum.therapist_text,
            patient_text=maximum.patient_text,
        )
    return aggregates


def project_to_phq(
    aggregates: dict[str, TopicAggregate],
    instrument: Instrument = DEFAULT_INSTRUMENT,
) -> dict[str, dict]:
    """Report each PHQ prior anchor against the topic that approximates it."""
    output_type = f"{instrument.name} proxy score"
    projected: dict[str, dict] = {}
    for symptom, topic in instrument.prior_topic_map.items():
        if topic is None:
            projected[symptom] = {
                "output_type": output_type,
                "topic": None,
                "mapping_quality": instrument.prior_mapping_quality[symptom],
                "raw_session_score": None,
                "rounded_session_score": None,
                "source_turn": None,
                "relevant_turns": [],
                "evidence": None,
                "number_of_relevant_turns": 0,
                "reason": f"No directly corresponding {instrument.name} topic",
            }
            continue

        aggregate = aggregates[topic]
        projected[symptom] = {
            "output_type": output_type,
            "topic": topic,
            "mapping_quality": instrument.prior_mapping_quality[symptom],
            "raw_session_score": aggregate.raw_session_score,
            "rounded_session_score": aggregate.rounded_session_score,
            "source_turn": aggregate.source_turn,
            "relevant_turns": aggregate.relevant_turns,
            "evidence": aggregate.evidence,
            "number_of_relevant_turns": aggregate.number_of_relevant_turns,
        }
    return projected
