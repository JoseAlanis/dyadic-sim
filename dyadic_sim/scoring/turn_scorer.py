"""Coordinate relevance, translation, formatting, and severity rating."""

from __future__ import annotations

from dyadic_sim.scoring.instrument import Instrument
from dyadic_sim.scoring.instruments import DEFAULT_INSTRUMENT
from dyadic_sim.scoring.types import TurnPair, TurnTopicResult


class TurnScorer:
    def __init__(
        self,
        relevance_detector,
        translator,
        regressor,
        instrument: Instrument = DEFAULT_INSTRUMENT,
    ):
        self.relevance_detector = relevance_detector
        self.translator = translator
        self.regressor = regressor
        self.instrument = instrument

    def score_session(self, pairs: list[TurnPair]) -> list[TurnTopicResult]:
        relevance_results = self.relevance_detector.detect_batch(pairs)
        pair_by_turn = {pair.turn_index: pair for pair in pairs}
        accepted = [result for result in relevance_results if result.accepted_for_scoring]

        texts_to_translate: list[str] = []
        for relevance in accepted:
            pair = pair_by_turn[relevance.turn_index]
            texts_to_translate.extend([pair.therapist_text, pair.patient_text])
            if relevance.evidence and relevance.evidence in pair.patient_text:
                texts_to_translate.append(relevance.evidence)
            if pair.earlier_therapist_text and pair.earlier_patient_text:
                texts_to_translate.extend(
                    [pair.earlier_therapist_text, pair.earlier_patient_text]
                )
        translations = self.translator.translate_batch(texts_to_translate)

        jobs: list[tuple[str, str]] = []
        job_details: dict[str, tuple[str, bool]] = {}
        for relevance in accepted:
            pair = pair_by_turn[relevance.turn_index]
            model_input, compacted = self.regressor.prepare_input(
                relevance.topic,
                translations[pair.therapist_text],
                translations[pair.patient_text],
                (
                    translations[pair.earlier_therapist_text]
                    if pair.earlier_therapist_text
                    else None
                ),
                (
                    translations[pair.earlier_patient_text]
                    if pair.earlier_patient_text
                    else None
                ),
                translations.get(relevance.evidence) if relevance.evidence else None,
            )
            job_id = f"{relevance.turn_index}:{relevance.topic}"
            jobs.append((job_id, model_input))
            job_details[job_id] = (model_input, compacted)

        predictions = self.regressor.predict_batch(jobs) if jobs else {}
        results: list[TurnTopicResult] = []
        for relevance in relevance_results:
            pair = pair_by_turn[relevance.turn_index]
            job_id = f"{relevance.turn_index}:{relevance.topic}"
            prediction = predictions.get(job_id)
            model_input, compacted = job_details.get(job_id, (None, False))
            results.append(
                TurnTopicResult(
                    turn_index=relevance.turn_index,
                    symptom=self.instrument.topic_to_prior_symptom.get(relevance.topic),
                    topic=relevance.topic,
                    relevant=relevance.relevant,
                    accepted_for_scoring=bool(prediction),
                    relevance_confidence=relevance.relevance_confidence,
                    relevance_method=relevance.relevance_method,
                    evidence=relevance.evidence,
                    therapist_text=pair.therapist_text,
                    patient_text=pair.patient_text,
                    translated_therapist_text=(
                        translations.get(pair.therapist_text) if prediction else None
                    ),
                    translated_patient_text=(
                        translations.get(pair.patient_text) if prediction else None
                    ),
                    raw_model_output=prediction.raw_model_output if prediction else None,
                    raw_score=prediction.raw_score if prediction else None,
                    rounded_score=prediction.rounded_score if prediction else None,
                    rejection_reason=(
                        None if prediction else relevance.rejection_reason
                    ),
                    rule_hit=relevance.rule_hit,
                    patient_similarity=relevance.patient_similarity,
                    combined_similarity=relevance.combined_similarity,
                    context_source=pair.context_source,
                    context_synthesized=pair.context_synthesized,
                    model_input=model_input,
                    input_compacted=compacted,
                )
            )
        return results
