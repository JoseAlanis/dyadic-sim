"""Typed records passed between symptom-scoring stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class JsonRecord:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SessionSource(JsonRecord):
    session_dir: Path
    metadata: dict[str, Any]
    transcript: list[dict[str, Any]]


@dataclass
class TurnPair(JsonRecord):
    turn_index: int
    therapist_text: str
    patient_text: str
    earlier_therapist_text: str | None = None
    earlier_patient_text: str | None = None
    context_source: str = "previous_transcript_row"
    context_synthesized: bool = False
    valid: bool = True
    exclusion_reason: str | None = None


@dataclass
class RelevanceResult(JsonRecord):
    turn_index: int
    topic: str
    relevant: bool
    relevance_confidence: float
    relevance_method: str
    evidence: str | None
    rule_hit: str | None = None
    patient_similarity: float | None = None
    combined_similarity: float | None = None
    accepted_for_scoring: bool = False
    rejection_reason: str | None = None


@dataclass
class ModelPrediction(JsonRecord):
    job_id: str
    raw_model_output: float
    raw_score: float
    rounded_score: int


@dataclass
class TurnTopicResult(JsonRecord):
    turn_index: int
    symptom: str | None
    topic: str
    relevant: bool
    accepted_for_scoring: bool
    relevance_confidence: float
    relevance_method: str
    evidence: str | None
    therapist_text: str
    patient_text: str
    translated_therapist_text: str | None = None
    translated_patient_text: str | None = None
    raw_model_output: float | None = None
    raw_score: float | None = None
    rounded_score: int | None = None
    rejection_reason: str | None = None
    rule_hit: str | None = None
    patient_similarity: float | None = None
    combined_similarity: float | None = None
    context_source: str = "previous_transcript_row"
    context_synthesized: bool = False
    model_input: str | None = None
    input_compacted: bool = False


@dataclass
class TopicAggregate(JsonRecord):
    topic: str
    raw_session_score: float | None
    rounded_session_score: int | None
    source_turn: int | None
    relevant_turns: list[int] = field(default_factory=list)
    evidence: str | None = None
    number_of_relevant_turns: int = 0
    therapist_text: str | None = None
    patient_text: str | None = None

