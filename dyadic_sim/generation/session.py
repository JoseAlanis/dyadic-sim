"""
dyadic_sim/generation/session.py

Session lifecycle:  start, resume, and close sessions.
Creates the session directory, metadata file, and transcript log.
Provides the append_turn() interface used by dyad.py.
"""

import hashlib
import json
import os
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

from dyadic_sim.generation.utterance import GENERATION_VERSION

SESSIONS_DIR = Path("data/sessions")

#: The exact prompt texts a session sent, by hash. Only ever grows.
PROMPTS_FILE = "prompts.json"

#: Version of what this module records about each completion (TurnRecord
#: diagnostics, PROMPTS_FILE, the metadata "record_diagnostics" block). Separate
#: from GENERATION_VERSION: recording more changes nothing that is generated.
DIAGNOSTICS_SCHEMA_VERSION = 1


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_prompt_store(session_dir: Path) -> list[dict] | None:
    """The validated entries of a session's PROMPTS_FILE; None if there is none.

    Raises ValueError naming the problem: unreadable JSON, another schema, an
    entry with missing or mistyped fields, a text that does not match its hash,
    or a duplicate entry.
    """
    path = session_dir / PROMPTS_FILE
    if not path.exists():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"{path} cannot be read ({error})") from error
    if (not isinstance(stored, dict)
            or stored.get("schema_version") != DIAGNOSTICS_SCHEMA_VERSION
            or not isinstance(stored.get("prompts"), list)):
        raise ValueError(f"{path} is not a schema {DIAGNOSTICS_SCHEMA_VERSION} prompt store")
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(stored["prompts"]):
        if not (isinstance(entry, dict)
                and all(isinstance(entry.get(k), str) for k in ("role", "sha256", "text"))
                and type(entry.get("first_turn")) is int):
            raise ValueError(f"{path}: entry {index} is malformed")
        if sha256_text(entry["text"]) != entry["sha256"]:
            raise ValueError(f"{path}: entry {index} ({entry['role']}) does not match its hash")
        key = (entry["role"], entry["sha256"])
        if key in seen:
            raise ValueError(f"{path}: entry {index} repeats an earlier entry")
        seen.add(key)
    return stored["prompts"]


def prompt_provenance_problem(session: "Session") -> str | None:
    """Why a session's recorded prompts cannot be trusted, or None.

    The mark and the rows must agree. A session without a mark (or with a null
    one) is one recorded before diagnostics existed: allowed only if no row
    carries diagnostics, and it needs no store, though one that exists must
    still be valid. A marked session's since_turn lies in 1..turn_count + 1, no
    row before it carries diagnostics, every row from it on carries both, and a
    valid store holds its opening and every system prompt those rows refer to.
    """
    try:
        entries = read_prompt_store(session.session_dir)
    except ValueError as error:
        return str(error)
    instrumented = [
        r.turn for r in session.transcript
        if r.patient_diagnostics is not None or r.therapist_diagnostics is not None
    ]
    mark = session.record_diagnostics
    if mark is None:
        if instrumented:
            return f"turn {instrumented[0]} carries diagnostics but the record_diagnostics mark is missing"
        return None
    if (not isinstance(mark, dict)
            or mark.get("schema_version") != DIAGNOSTICS_SCHEMA_VERSION
            or type(mark.get("since_turn")) is not int):
        return f"its record_diagnostics mark is not schema {DIAGNOSTICS_SCHEMA_VERSION}"
    since = mark["since_turn"]
    if not 1 <= since <= session.turn_count + 1:
        return (f"its record_diagnostics mark starts at turn {since}, outside "
                f"1..{session.turn_count + 1}")
    if instrumented and instrumented[0] < since:
        return f"turn {instrumented[0]} carries diagnostics before the mark (turn {since})"
    if entries is None:
        return f"{PROMPTS_FILE} is missing"
    stored = {(e["role"], e["sha256"]) for e in entries}
    opening = session.initial_patient_prompt
    if opening and ("opening", sha256_text(opening)) not in stored:
        return f"its opening is not in {PROMPTS_FILE}"
    for record in session.transcript:
        if record.turn < since:
            continue
        for role in ("patient", "therapist"):
            diagnostics = getattr(record, f"{role}_diagnostics")
            if not isinstance(diagnostics, dict):
                return f"turn {record.turn} has no {role} diagnostics"
            if (f"{role}_system", diagnostics.get("system_prompt_sha256")) not in stored:
                return f"the {role} prompt of turn {record.turn} is not in {PROMPTS_FILE}"
    return None


def completion_record(
    role: str,
    turn: int,
    system_prompt_sha256: str,
    messages: list,
    response,
    elapsed_s: float,
) -> dict:
    """What one completion was sent and what came back, for TurnRecord.

    `messages` are the Message objects sent; they are hashed, not stored, since
    the transcript and the stored opening rebuild them. Everything taken from
    `response` is copied as reported, so None stays None (unknown), never 0.
    """
    sent = json.dumps([[m.role, m.content] for m in messages], ensure_ascii=False)
    return {
        "role": role,
        "turn": turn,
        "system_prompt_sha256": system_prompt_sha256,
        "messages_sha256": sha256_text(sent),
        "n_messages": len(messages),
        "provider": response.provider,
        "requested": response.request,
        "observed": response.observed,
        "finish_reason": response.finish_reason,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "elapsed_s": round(elapsed_s, 3),
    }


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
    # completion_record() for each agent's call on this turn. None on turns
    # recorded before diagnostics existed: unknown, not empty. Saved with the
    # completed turn, so a turn that fails (say, the therapist call errors after
    # the patient call returned) leaves no diagnostics for either call; a
    # journal of every call would be separate work.
    patient_diagnostics: dict | None = None
    therapist_diagnostics: dict | None = None

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
            patient_prompt_source: str | None = None,
            record_diagnostics: dict | None = None,
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
        # Where the patient prompt came from when it was not the case file named
        # by case_name (None). Read from the record; written only through
        # save_metadata(extra), so a caller's more specific label is kept.
        self.patient_prompt_source = patient_prompt_source
        # The metadata "record_diagnostics" mark (see begin_diagnostics); None
        # until diagnostics start, and on records from before they existed.
        self.record_diagnostics = record_diagnostics
        self.turn_count = 0
        self.transcript: list[TurnRecord] = []
        self.created_at = datetime.now(UTC).replace(tzinfo=None).isoformat()

        self.session_dir = SESSIONS_DIR / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self._transcript_path = self.session_dir / "transcript.jsonl"
        self._metadata_path = self.session_dir / "metadata.json"
        self._prompts_path = self.session_dir / PROMPTS_FILE
        self._prompts: list[dict] | None = None  # PROMPTS_FILE, read on first use
        self._pdf_path = self.session_dir / self._build_pdf_filename()

    def begin_diagnostics(self, first_turn: int) -> None:
        """Mark the turn from which this record carries completion diagnostics.

        A new session starts at turn 1. A session resumed from before
        diagnostics existed starts at its first new turn, so its earlier turns
        stay unknown instead of inheriting today's settings. A mark already in
        the metadata is never moved.
        """
        stored = self._read_metadata().get("record_diagnostics")
        if stored is not None:
            self.record_diagnostics = stored
            return
        self.record_diagnostics = {
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "since_turn": first_turn,
            "prompts_file": PROMPTS_FILE,
        }
        self.save_metadata({"record_diagnostics": self.record_diagnostics})

    def record_prompt(self, role: str, text: str, turn: int) -> str:
        """Keep the exact text of a prompt this session sends; return its hash.

        PROMPTS_FILE only grows. A text already stored for `role` is not added
        again, and stored entries are never changed, so a resumed run cannot
        rewrite what an earlier run sent; a prompt that changed between runs
        appears as a new entry with the turn it was first used.
        """
        digest = sha256_text(text)
        if self._prompts is None:
            self._prompts = self._read_prompts()
        if any(e["role"] == role and e["sha256"] == digest for e in self._prompts):
            return digest
        updated = [*self._prompts, {
            "role": role,
            "sha256": digest,
            "first_turn": turn,
            "recorded_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
            "text": text,
        }]
        # Written whole and swapped in, so a crash leaves the old file intact.
        # The in-memory copy follows only once the file is replaced: if the
        # write fails, a retry must write this prompt again, not find it cached.
        partial = self._prompts_path.with_suffix(".json.partial")
        try:
            partial.write_text(
                json.dumps({"schema_version": DIAGNOSTICS_SCHEMA_VERSION,
                            "prompts": updated}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(partial, self._prompts_path)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        self._prompts = updated
        return digest

    def _read_prompts(self) -> list[dict]:
        """The stored prompts. An invalid file is an error, never overwritten."""
        try:
            return read_prompt_store(self.session_dir) or []
        except ValueError as error:
            raise RuntimeError(
                f"{error}; refusing to overwrite the prompts this session already sent."
            ) from error

    def _read_metadata(self) -> dict:
        try:
            with open(self._metadata_path, encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def append_turn(
        self,
        therapist_text: str,
        patient_text: str,
        therapist_tokens: int | None = None,
        patient_tokens: int | None = None,
        symptom_discussion_started: bool = False,
        therapist_text_raw: str | None = None,
        patient_text_raw: str | None = None,
        patient_diagnostics: dict | None = None,
        therapist_diagnostics: dict | None = None,
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
            patient_diagnostics:          completion_record() of the patient call
            therapist_diagnostics:        completion_record() of the therapist call

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
            patient_diagnostics=patient_diagnostics,
            therapist_diagnostics=therapist_diagnostics,
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
        existing = self._read_metadata()

        meta = {
            "metadata_schema_version": 2,
            # What the agents were told to produce and what was done to their
            # output; see dyadic_sim/generation/utterance.py. Absent means version 1.
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
        patient_prompt_source=meta.get("patient_prompt_source"),
        record_diagnostics=meta.get("record_diagnostics"),
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


def check_continuable(session: Session, prior_supplied: bool = False) -> None:
    """Refuse to continue a session that the current generator did not start.

    Appending turns under another generation contract would mix two stimuli in
    one transcript. Records from other versions stay readable (resume_session,
    the scoring pipeline, the analyses); they only cannot grow. The same holds
    for the patient prompt: a session started from a supplied prior continues
    only when the caller supplies a prior again, and a session built from its
    case file only when it does not. And a session that records diagnostics
    continues only while its stored prompts account for every recorded turn
    (prompt_provenance_problem). Call before any agent is built or anything is
    written.
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
    if bool(session.patient_prompt_source) != prior_supplied:
        if prior_supplied:
            raise IncompatibleSessionError(
                f"{session.session_id} was built from case {session.case_name!r}; "
                "it cannot continue with a supplied patient prior."
            )
        raise IncompatibleSessionError(
            f"{session.session_id} was generated from a supplied patient prompt "
            f"({session.patient_prompt_source}), not from case {session.case_name!r}; "
            "continue it with the runner that supplied it."
        )
    problem = prompt_provenance_problem(session)
    if problem:
        raise IncompatibleSessionError(
            f"{session.session_id} cannot be continued: {problem}, so the prompts "
            "behind its recorded turns are not accounted for."
        )

