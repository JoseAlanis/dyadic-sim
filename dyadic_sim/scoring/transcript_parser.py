"""Load saved sessions and construct correctly ordered dialogue pairs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dyadic_sim.generation.opening import V1_OPENING
from dyadic_sim.scoring.config import PHQ_SYMPTOM_LABELS
from dyadic_sim.scoring.types import SessionSource, TurnPair

_REFERENTIAL_REPLY = re.compile(
    r"^\s*(?:[\(\[].*?[\)\]]\s*)?"
    r"(?:yes|yeah|yep|no|nope|sometimes|often|always|rarely|never|"
    r"that\b|this\b|it\b|they\b|he\b|she\b|worse\b|better\b)",
    re.IGNORECASE | re.DOTALL,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                # ValueError, not TypeError: this is a malformed file, the same
                # kind of failure as the decode error above, not a bad argument.
                raise ValueError(f"Expected a JSON object on {path}:{line_number}")  # noqa: TRY004
            rows.append(row)
    return rows


def load_session_source(path: str | Path) -> SessionSource:
    source_path = Path(path)
    if source_path.is_dir():
        session_dir = source_path
        transcript_path = session_dir / "transcript.jsonl"
    else:
        transcript_path = source_path
        session_dir = transcript_path.parent

    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    metadata_path = session_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise ValueError(f"Metadata must be a JSON object: {metadata_path}")

    transcript = read_jsonl(transcript_path)
    if not transcript:
        raise ValueError(f"Transcript contains no turns: {transcript_path}")

    metadata.setdefault("session_id", session_dir.name)
    return SessionSource(session_dir=session_dir, metadata=metadata, transcript=transcript)


def needs_reference_context(patient_text: str) -> bool:
    plain = patient_text.strip()
    word_count = len(re.findall(r"\b\w+\b", plain))
    return word_count <= 8 or bool(_REFERENTIAL_REPLY.search(plain))


def detect_echo_artifact(
    patient_text: str,
    preceding_therapist_text: str,
) -> bool:
    patient = " ".join(patient_text.split()).casefold()
    therapist = " ".join(preceding_therapist_text.split()).casefold()
    return bool(patient and therapist and patient == therapist)


def build_turn_pairs(
    source: SessionSource,
    legacy_prompt_fallback: str | None = V1_OPENING,
) -> list[TurnPair]:
    """Pair each patient row with the therapist message from the previous row."""

    rows = source.transcript
    initial_prompt = source.metadata.get("initial_patient_prompt")
    pairs: list[TurnPair] = []

    for index, row in enumerate(rows):
        patient_text = str(row.get("patient_text") or "").strip()
        turn_index = int(row.get("turn", index + 1))
        context_source = "previous_transcript_row"
        context_synthesized = False

        if index == 0:
            if initial_prompt:
                therapist_text = str(initial_prompt).strip()
                context_source = "metadata_initial_patient_prompt"
            elif legacy_prompt_fallback:
                therapist_text = legacy_prompt_fallback
                context_source = "legacy_known_opening_prompt"
                context_synthesized = True
            else:
                therapist_text = ""
                context_source = "missing_initial_patient_prompt"
        else:
            therapist_text = str(rows[index - 1].get("therapist_text") or "").strip()

        valid = True
        exclusion_reason = None
        if not patient_text:
            valid = False
            exclusion_reason = "empty_patient_text"
        elif not therapist_text:
            valid = False
            exclusion_reason = "missing_preceding_therapist_text"
        elif detect_echo_artifact(patient_text, therapist_text):
            valid = False
            exclusion_reason = "verbatim_echo_artifact"

        earlier_therapist = None
        earlier_patient = None
        if valid and index >= 2 and needs_reference_context(patient_text):
            earlier_therapist = str(rows[index - 2].get("therapist_text") or "").strip() or None
            earlier_patient = str(rows[index - 1].get("patient_text") or "").strip() or None

        pairs.append(
            TurnPair(
                turn_index=turn_index,
                therapist_text=therapist_text,
                patient_text=patient_text,
                earlier_therapist_text=earlier_therapist,
                earlier_patient_text=earlier_patient,
                context_source=context_source,
                context_synthesized=context_synthesized,
                valid=valid,
                exclusion_reason=exclusion_reason,
            )
        )

    return pairs


def load_prior_levels(metadata: dict[str, Any]) -> tuple[dict[str, str], str]:
    raw = metadata.get("patient_symptom_levels")
    if isinstance(raw, dict):
        return ({str(key): str(value) for key, value in raw.items()}, "metadata_raw")

    legacy = str(metadata.get("patient_symptoms") or "")
    if legacy:
        label_to_key = {label.casefold(): key for key, label in PHQ_SYMPTOM_LABELS.items()}
        recovered: dict[str, str] = {}
        frequencies = (
            "not at all",
            "several days",
            "more than half the days",
            "nearly every day",
        )
        for line in legacy.splitlines():
            clean = line.strip().lstrip("-").strip()
            if ":" not in clean:
                continue
            label, remainder = clean.split(":", 1)
            key = label_to_key.get(label.strip().casefold())
            if not key:
                continue
            lower = remainder.casefold()
            frequency = next((item for item in frequencies if lower.startswith(item)), None)
            if frequency:
                recovered[key] = frequency
        if recovered:
            return recovered, "metadata_legacy_formatted"

    case_name = str(metadata.get("case_name") or "")
    match = re.search(
        r"_(lack_of_pleasure|depressed_mood|sleep_problems|low_energy|"
        r"appetite_changes|feelings_of_failure_or_guilt|concentration_problems|"
        r"psychomotor_changes|thoughts_of_death_or_self_harm)_run_\d+$",
        case_name,
    )
    if match:
        return {match.group(1): "nearly every day"}, "case_name_inferred_incomplete"

    return {}, "unavailable"
