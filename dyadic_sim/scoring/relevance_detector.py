"""Auditable relevance detection, run before the rater sees a turn."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from dyadic_sim.scoring.instrument import Instrument
from dyadic_sim.scoring.instruments import DEFAULT_INSTRUMENT
from dyadic_sim.scoring.types import RelevanceResult, TurnPair

_REFERENCE_ANSWER = re.compile(
    r"^\s*(?:[\(\[].*?[\)\]]\s*)?"
    r"(?:yes|yeah|yep|no|nope|not really|sometimes|often|always|rarely|never|"
    r"that|this|it)\b",
    re.IGNORECASE | re.DOTALL,
)


def _evidence_sentence(text: str, pattern: re.Pattern[str] | None = None) -> str:
    sentences = [
        chunk.strip()
        for chunk in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        if chunk.strip()
    ]
    if not sentences:
        return text.strip()
    if pattern:
        for sentence in sentences:
            if pattern.search(sentence):
                return sentence
    return sentences[0]


class HybridRelevanceDetector:
    """Rules first, with an optional sentence-embedding semantic fallback."""

    def __init__(
        self,
        *,
        instrument: Instrument = DEFAULT_INSTRUMENT,
        mode: str = "hybrid",
        combined_threshold: float = 0.50,
        patient_threshold: float = 0.35,
        acceptance_threshold: float = 0.80,
        embedder: Any | None = None,
    ):
        if mode not in {"hybrid", "rules"}:
            raise ValueError("Relevance mode must be 'hybrid' or 'rules'")
        self.instrument = instrument
        self.mode = mode
        self.combined_threshold = combined_threshold
        self.patient_threshold = patient_threshold
        self.acceptance_threshold = acceptance_threshold
        self._embedder = embedder
        self._patterns = {
            topic: re.compile(instrument.relevance_patterns[topic], re.IGNORECASE)
            for topic in instrument.topics
        }

    def _get_embedder(self):
        if self._embedder is None:
            from dyadic_sim.analysis.embeddings import get_model

            self._embedder = get_model()
        return self._embedder

    def detect_batch(self, pairs: list[TurnPair]) -> list[RelevanceResult]:
        patient_sims: dict[tuple[int, str], float] = {}
        combined_sims: dict[tuple[int, str], float] = {}

        topics = self.instrument.topics
        valid_pairs = [pair for pair in pairs if pair.valid]
        if self.mode == "hybrid" and valid_pairs:
            embedder = self._get_embedder()
            topic_vectors = np.asarray(
                embedder.encode(
                    [self.instrument.topic_descriptions[topic] for topic in topics],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )
            patient_vectors = np.asarray(
                embedder.encode(
                    [pair.patient_text for pair in valid_pairs],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )
            combined_vectors = np.asarray(
                embedder.encode(
                    [
                        f"Therapist: {pair.therapist_text}\nPatient: {pair.patient_text}"
                        for pair in valid_pairs
                    ],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )
            patient_matrix = patient_vectors @ topic_vectors.T
            combined_matrix = combined_vectors @ topic_vectors.T
            for pair_index, pair in enumerate(valid_pairs):
                for topic_index, topic in enumerate(topics):
                    key = (pair.turn_index, topic)
                    patient_sims[key] = float(patient_matrix[pair_index, topic_index])
                    combined_sims[key] = float(combined_matrix[pair_index, topic_index])

        results: list[RelevanceResult] = []
        for pair in pairs:
            for topic in topics:
                if not pair.valid:
                    results.append(
                        RelevanceResult(
                            turn_index=pair.turn_index,
                            topic=topic,
                            relevant=False,
                            relevance_confidence=1.0,
                            relevance_method="invalid_turn",
                            evidence=None,
                            rejection_reason=pair.exclusion_reason,
                        )
                    )
                    continue

                pattern = self._patterns[topic]
                patient_match = pattern.search(pair.patient_text)
                therapist_match = pattern.search(pair.therapist_text)
                key = (pair.turn_index, topic)
                patient_similarity = patient_sims.get(key)
                combined_similarity = combined_sims.get(key)

                if patient_match:
                    evidence = _evidence_sentence(pair.patient_text, pattern)
                    result = RelevanceResult(
                        turn_index=pair.turn_index,
                        topic=topic,
                        relevant=True,
                        relevance_confidence=0.95,
                        relevance_method="patient_rule",
                        evidence=evidence,
                        rule_hit=patient_match.group(0),
                        patient_similarity=patient_similarity,
                        combined_similarity=combined_similarity,
                    )
                elif therapist_match and _REFERENCE_ANSWER.search(pair.patient_text):
                    evidence = (
                        f"Therapist: {_evidence_sentence(pair.therapist_text, pattern)} "
                        f"Patient: {_evidence_sentence(pair.patient_text)}"
                    )
                    result = RelevanceResult(
                        turn_index=pair.turn_index,
                        topic=topic,
                        relevant=True,
                        relevance_confidence=0.90,
                        relevance_method="therapist_rule_referential_answer",
                        evidence=evidence,
                        rule_hit=therapist_match.group(0),
                        patient_similarity=patient_similarity,
                        combined_similarity=combined_similarity,
                    )
                elif (
                    self.mode == "hybrid"
                    and patient_similarity is not None
                    and combined_similarity is not None
                    and patient_similarity >= self.patient_threshold
                    and combined_similarity >= self.combined_threshold
                ):
                    margin = min(
                        patient_similarity - self.patient_threshold,
                        combined_similarity - self.combined_threshold,
                    )
                    confidence = min(0.99, 0.80 + max(0.0, margin) * 0.38)
                    result = RelevanceResult(
                        turn_index=pair.turn_index,
                        topic=topic,
                        relevant=True,
                        relevance_confidence=confidence,
                        relevance_method="semantic_similarity",
                        evidence=_evidence_sentence(pair.patient_text),
                        patient_similarity=patient_similarity,
                        combined_similarity=combined_similarity,
                    )
                else:
                    result = RelevanceResult(
                        turn_index=pair.turn_index,
                        topic=topic,
                        relevant=False,
                        relevance_confidence=0.95 if not therapist_match else 0.70,
                        relevance_method=(
                            "rules_no_match" if self.mode == "rules" else "below_semantic_threshold"
                        ),
                        evidence=None,
                        rule_hit=therapist_match.group(0) if therapist_match else None,
                        patient_similarity=patient_similarity,
                        combined_similarity=combined_similarity,
                        rejection_reason="insufficient_relevance_evidence",
                    )

                result.accepted_for_scoring = bool(
                    result.relevant
                    and result.evidence
                    and result.relevance_confidence >= self.acceptance_threshold
                )
                if result.relevant and not result.accepted_for_scoring:
                    result.rejection_reason = "below_relevance_acceptance_threshold"
                results.append(result)

        return results

