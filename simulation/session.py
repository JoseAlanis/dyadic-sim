"""
simulation/session.py

Session lifecycle:  start, resume, and close sessions.
Creates the session directory, metadata file, and transcript log.
Provides the append_turn() interface used by dyad.py.
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from simulation.utterance import GENERATION_VERSION

SESSIONS_DIR = Path("data/sessions")


@dataclass
class TurnRecord:
    """One complete turn in a session transcript."""
    turn: int
    therapist_text: str
    patient_text: str
    therapist_model: str
    patient_model: str
    therapist_tokens: int | None
    patient_tokens: int | None
    symptom_discussion_started: bool = False
    # Pre-cleaning text, recorded only when clean_utterance() changed something
    # (stage directions, emphasis spans, or a leaked role label were removed).
    therapist_text_raw: str | None = None
    patient_text_raw: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).replace(tzinfo=None).isoformat())

    def to_dict(self) -> dict:
        return self.__dict__.copy()


#: Fields a replayed transcript row is allowed to populate. Older transcripts
#: carry fields that later versions dropped; replaying them must not fail.
_TURN_FIELDS = frozenset(TurnRecord.__dataclass_fields__)


class Session:
    """
    Manages a single simulation session (directory, transcript, metadata).
    """

    def __init__(
            self,
            session_id: str,
            therapist_model: str,
            patient_model: str,
            case_name: str,
            orientation: str,
            patient_symptoms: str = "",
            patient_symptom_levels: dict[str, str] | None = None,
            patient_id: str | None = None,
            conversation_language: str = "en",
            initial_patient_prompt: str | None = None,
            generation_version: int | None = GENERATION_VERSION,
    ):
        self.session_id = session_id
        self.therapist_model = therapist_model
        self.patient_model = patient_model
        self.case_name = case_name
        self.orientation = orientation
        self.patient_symptoms = patient_symptoms
        self.patient_symptom_levels = patient_symptom_levels or {}
        self.patient_id = patient_id
        self.conversation_language = conversation_language
        self.initial_patient_prompt = initial_patient_prompt
        # The generator that wrote this record. A new session gets the current
        # one; a resumed session keeps what its metadata says (None = absent =
        # version 1), and saving never overwrites it.
        self.generation_version = generation_version
        self.turn_count = 0
        self.transcript: list[TurnRecord] = []
        self.created_at = datetime.now(UTC).replace(tzinfo=None).isoformat()

        self.session_dir = SESSIONS_DIR / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self._transcript_path = self.session_dir / "transcript.jsonl"
        self._metadata_path = self.session_dir / "metadata.json"
        self._pdf_path = self.session_dir / self._build_pdf_filename()

    def append_turn(
        self,
        therapist_text: str,
        patient_text: str,
        therapist_tokens: int | None = None,
        patient_tokens: int | None = None,
        symptom_discussion_started: bool = False,
        therapist_text_raw: str | None = None,
        patient_text_raw: str | None = None,
    ) -> TurnRecord:
        """
        Record a completed turn to the transcript.

        Args:
            therapist_text:               What the therapist said
            patient_text:                 What the patient said
            therapist_tokens:             Token count for therapist response
            patient_tokens:               Token count for patient response
            symptom_discussion_started:   Whether the patient began explicitly
                                          talking about symptoms on this turn
            therapist_text_raw:           Pre-cleaning therapist text, passed only
                                          when cleaning removed something
            patient_text_raw:             Pre-cleaning patient text, same

        Returns:
            The TurnRecord appended to the transcript
        """
        self.turn_count += 1
        record = TurnRecord(
            turn=self.turn_count,
            therapist_text=therapist_text,
            patient_text=patient_text,
            therapist_model=self.therapist_model,
            patient_model=self.patient_model,
            therapist_tokens=therapist_tokens,
            patient_tokens=patient_tokens,
            symptom_discussion_started=symptom_discussion_started,
            therapist_text_raw=therapist_text_raw,
            patient_text_raw=patient_text_raw,
        )
        self.transcript.append(record)

        # Append to JSONL file immediately (survive crashes)
        with open(self._transcript_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

        # Also update the human-readable PDF transcript.
        self.save_pdf_transcript()

        return record

    def _pdf_safe_text(self, text: str | None) -> str:
        """
        Make text safe for ReportLab Paragraphs.

        ReportLab Paragraph uses a small XML-like markup language, so we escape
        characters such as <, >, and &. We also preserve line breaks.
        """
        if not text:
            return ""

        return escape(str(text)).replace("\n", "<br/>")

    def save_pdf_transcript(self) -> None:
        """
        Generate a readable PDF version of the session transcript.

        This rebuilds the PDF from self.transcript each time it is called.
        That is safer than trying to append directly to an existing PDF.
        """
        doc = SimpleDocTemplate(
            str(self._pdf_path),
            pagesize=A4,
            rightMargin=1.7 * cm,
            leftMargin=1.7 * cm,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm,
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "TranscriptTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=18,
            leading=22,
            spaceAfter=14,
        )

        meta_style = ParagraphStyle(
            "TranscriptMeta",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.darkgrey,
            spaceAfter=4,
        )

        turn_style = ParagraphStyle(
            "TurnHeading",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.HexColor("#333333"),
        )

        speaker_style = ParagraphStyle(
            "Speaker",
            parent=styles["Heading3"],
            fontSize=11,
            leading=14,
            spaceBefore=6,
            spaceAfter=3,
            textColor=colors.HexColor("#222222"),
        )

        body_style = ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontSize=10,
            leading=14,
            spaceAfter=8,
        )

        story = []

        story.append(Paragraph("Dyadic Therapy Simulation Transcript", title_style))

        story.append(Paragraph(f"<b>Session ID:</b> {self._pdf_safe_text(self.session_id)}", meta_style))
        story.append(Paragraph(f"<b>Case:</b> {self._pdf_safe_text(self.case_name)}", meta_style))
        story.append(Paragraph(f"<b>Orientation:</b> {self._pdf_safe_text(self.orientation)}", meta_style))
        story.append(Paragraph(f"<b>Therapist model:</b> {self._pdf_safe_text(self.therapist_model)}", meta_style))
        story.append(Paragraph(f"<b>Patient model:</b> {self._pdf_safe_text(self.patient_model)}", meta_style))
        story.append(Paragraph(f"<b>Created at:</b> {self._pdf_safe_text(self.created_at)}", meta_style))
        story.append(Paragraph(f"<b>Total turns:</b> {self.turn_count}", meta_style))

        story.append(Spacer(1, 12))

        if self.patient_symptoms.strip():
            story.append(Paragraph("Patient Symptom Profile", turn_style))
            story.append(
                Paragraph(
                    self._pdf_safe_text(self.patient_symptoms),
                    body_style,
                )
            )
            story.append(Spacer(1, 12))

        if not self.transcript:
            story.append(Paragraph("No turns have been recorded yet.", body_style))
        else:
            for record in self.transcript:
                turn_block = [
                    Paragraph(f"Turn {record.turn}", turn_style),
                    Paragraph("Patient", speaker_style),
                    Paragraph(self._pdf_safe_text(record.patient_text), body_style),
                    Paragraph("Therapist", speaker_style),
                    Paragraph(
                        self._pdf_safe_text(record.therapist_text)
                        if record.therapist_text
                        else "<i>No therapist response recorded.</i>",
                        body_style,
                    ),
                ]

                story.append(KeepTogether(turn_block))
                story.append(Spacer(1, 8))

        doc.build(story)



    def save_metadata(self, extra: dict | None = None) -> None:
        """Write / update session metadata.

        Merges into whatever is already on disk, so successive calls with
        different `extra` keys accumulate instead of overwriting each other.
        """
        existing: dict = {}
        if self._metadata_path.exists():
            try:
                with open(self._metadata_path, encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    existing = loaded
            except (OSError, json.JSONDecodeError):
                existing = {}

        meta = {
            "metadata_schema_version": 2,
            # What the agents were told to produce and what was done to their
            # output; see simulation/utterance.py. Absent == version 1 (pilot).
            "generation_version": self.generation_version,
            "session_id": self.session_id,
            "patient_id": self.patient_id,
            "therapist_model": self.therapist_model,
            "patient_model": self.patient_model,
            "case_name": self.case_name,
            "orientation": self.orientation,
            "patient_symptoms": self.patient_symptoms,
            "patient_symptom_levels": self.patient_symptom_levels,
            "conversation_language": self.conversation_language,
            "initial_patient_prompt": self.initial_patient_prompt,
            "turn_count": self.turn_count,
            "created_at": self.created_at,
            "updated_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
            "pdf_transcript_path": str(self._pdf_path),
        }

        if self.generation_version is None:
            del meta["generation_version"]  # keep an unversioned record unversioned
        merged = {**existing, **meta}
        if extra:
            merged.update(extra)
        meta = merged

        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def get_history(self) -> list[tuple[str, str]]:
        """
        Return transcript as list of (patient_text, therapist_text) pairs.
        Used by turn_manager to build message history.
        """
        return [(r.patient_text, r.therapist_text) for r in self.transcript]

    def _safe_filename_part(self, value: str) -> str:
        """
        Convert text into a safe filename part.

        Example:
            llama3.1 -> llama3_1
            afraid/of/dogs -> afraid_of_dogs
        """
        value = str(value).strip().lower()
        value = re.sub(r"[^a-zA-Z0-9]+", "_", value)
        value = value.strip("_")
        return value or "unknown"

    def _build_pdf_filename(self) -> str:
        """
        Build a readable PDF filename for this session transcript.
        """
        case = self._safe_filename_part(self.case_name)
        orientation = self._safe_filename_part(self.orientation)
        therapist = self._safe_filename_part(self.therapist_model)
        patient = self._safe_filename_part(self.patient_model)

        return f"{case}_{orientation}_{therapist}_vs_{patient}_transcript.pdf"


def new_session(
    therapist_model: str,
    patient_model: str,
    case_name: str,
    orientation: str,
    patient_symptoms: str = "",
    patient_symptom_levels: dict[str, str] | None = None,
    patient_id: str | None = None,
    conversation_language: str = "en",
    initial_patient_prompt: str | None = None,
) -> Session:
    """Create a new session with a fresh ID."""
    session_id = f"session_{datetime.now(UTC).replace(tzinfo=None).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    session = Session(
        session_id=session_id,
        therapist_model=therapist_model,
        patient_model=patient_model,
        case_name=case_name,
        orientation=orientation,
        patient_symptoms=patient_symptoms,
        patient_symptom_levels=patient_symptom_levels,
        patient_id=patient_id,
        conversation_language=conversation_language,
        initial_patient_prompt=initial_patient_prompt,
    )

    session.save_metadata()
    return session


def resume_session(session_id: str) -> Session:
    """
    Resume an existing session from its directory.
    Replays the transcript to restore turn_count and history.
    """
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")

    meta_path = session_dir / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    session = Session(
        session_id=session_id,
        therapist_model=meta["therapist_model"],
        patient_model=meta["patient_model"],
        case_name=meta["case_name"],
        orientation=meta["orientation"],
        patient_symptoms=meta.get("patient_symptoms", ""),
        patient_symptom_levels=meta.get("patient_symptom_levels", {}),
        patient_id=meta.get("patient_id"),
        conversation_language=meta.get("conversation_language", "en"),
        initial_patient_prompt=meta.get("initial_patient_prompt"),
        generation_version=meta.get("generation_version"),
    )

    # Replay transcript
    transcript_path = session_dir / "transcript.jsonl"
    if transcript_path.exists():
        with open(transcript_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    record = TurnRecord(
                        **{k: v for k, v in data.items() if k in _TURN_FIELDS}
                    )
                    session.transcript.append(record)
                    session.turn_count = record.turn

    session.created_at = meta.get("created_at", session.created_at)
    return session


class IncompatibleSessionError(RuntimeError):
    """A saved session that the current generator must not continue."""


def check_continuable(session: Session) -> None:
    """Refuse to continue a session that the current generator did not start.

    Appending turns under another generation contract would mix two stimuli in
    one transcript. Records from other versions stay readable (resume_session,
    the scoring pipeline, the analyses); they only cannot grow. Call before any
    agent is built or anything is written.
    """
    recorded = session.generation_version
    if recorded != GENERATION_VERSION:
        shown = "1 (implicit)" if recorded is None else recorded
        raise IncompatibleSessionError(
            f"{session.session_id} was generated at generation_version {shown}; "
            f"this generator is version {GENERATION_VERSION} and cannot continue "
            "it. Start a new session instead."
        )
    if not session.initial_patient_prompt:
        raise IncompatibleSessionError(
            f"{session.session_id} has no saved opening (initial_patient_prompt), "
            "so its conversation cannot be rebuilt."
        )

