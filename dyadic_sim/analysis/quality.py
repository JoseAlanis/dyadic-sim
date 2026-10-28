"""Frozen quality-control flags for generated transcripts.

Two data-quality problems that are not analysis bugs
but distort both the lexicon and the embedding if left unmarked:

**Role confusion.** Some generated patient turns are written in therapist
voice - empathic validation followed by clinical questions put to the therapist.
The loaders correctly attribute it to `patient_text`, so structurally it is
patient speech, but semantically it is not patient self-report. Scoring it as
symptom expression measures the generator's role compliance, not the patient's
symptoms.

**Annotation-only turns.** `dyadic_sim.generation.utterance.clean_utterance` deliberately
falls back to the original text when cleaning would empty a turn, because at
GENERATION time an empty turn would be written into the transcript and break the
dialogue. The consequence is that `clean=True` can still hand the analysis
"(pauses)" - a turn that is entirely stage direction. That belongs excluded here
rather than fixed there.

Both are reported, never silently applied. Exclusion is a pre-specified analysis
choice, and the rate by condition is itself worth knowing: if role confusion
correlates with condition, it is a confound rather than noise.
"""

from __future__ import annotations

import re

from dyadic_sim.generation.utterance import clean_utterance

# Therapist moves appearing in a patient turn. Deliberately narrow: these are
# clinician speech-acts, not merely empathic or questioning language, because
# patients ask questions too ("is that normal?") and a loose rule would flag
# ordinary dialogue.
_THERAPIST_MOVES = (
    r"\bcan you (?:tell me more|say more|describe|walk me through)\b",
    r"\bwhat (?:do you think|comes up|does that bring up|would it be like)\b",
    r"\bhow (?:does that (?:make you )?feel|do you feel about that)\b",
    r"\bit sounds like you(?:'re| are)\b",
    r"\bthat sounds (?:really |very )?(?:difficult|hard|painful|challenging)\b",
    r"\bwhat i'?m hearing is\b",
    r"\bthank you for (?:sharing|telling me|trusting)\b",
    r"\bwould you be (?:open to|willing to)\b",
    r"\blet'?s (?:explore|unpack|look at) that\b",
    r"\bhow (?:has|have) (?:that|this|these) been (?:affecting|impacting) you\b",
    r"\bin our (?:next )?session\b",
    r"\bas your therapist\b",
)
_THERAPIST_MOVE_RE = re.compile("|".join(_THERAPIST_MOVES), re.IGNORECASE)

# Bracketed narration or emphasis spans, i.e. what clean_utterance strips.
_ANNOTATION_RE = re.compile(r"[(\[][^)\]]*[)\]]|\*[^*]+\*")


def is_annotation_only(text: str) -> bool:
    """True when a turn is entirely stage direction, e.g. "(pauses)".

    Detected by comparing against `clean_utterance`'s own output rather than by a
    second stripping rule, so the two cannot drift apart: if cleaning removes
    everything and falls back to the original, what remains after removing
    bracketed spans is empty.
    """
    raw = (text or "").strip()
    if not raw:
        return True
    if clean_utterance(raw).strip() != raw:
        # cleaning changed something, so it did not need the fallback
        return False
    return not _ANNOTATION_RE.sub("", raw).strip()


def role_confusion_moves(text: str) -> list[str]:
    """Therapist speech-acts found in this (patient) turn."""
    return [m.group(0) for m in _THERAPIST_MOVE_RE.finditer(text or "")]


def flag_session(turns: list[tuple[int, str]]) -> dict:
    """QC summary for one session's patient turns.

    `turns` is what `sessions.patient_turns` returns.
    """
    confused, annotation_only = [], []
    for index, text in turns:
        if is_annotation_only(text):
            annotation_only.append(index)
        elif role_confusion_moves(text):
            confused.append(index)
    return {
        "n_turns": len(turns),
        "role_confused_turns": confused,
        "annotation_only_turns": annotation_only,
        "any_role_confusion": bool(confused),
    }


def qc_fields(turns: list[tuple[int, str]], meta: dict | None = None) -> dict:
    """Flat, CSV-ready QC record for one session.

    Every analysis writes these columns, so a result row carries its own quality
    context and a reader never has to trust that flagged content was handled
    somewhere upstream. `quality.py` previously existed but nothing called it,
    which is the same as not having it.

    Turn indices are stored as `;`-joined strings rather than lists so they
    survive a CSV round-trip without being re-parsed as text.
    """
    meta = meta or {}
    flags = flag_session(turns)
    return {
        "n_patient_turns": flags["n_turns"],
        "role_confused_turns": ";".join(str(i) for i in flags["role_confused_turns"]),
        "annotation_only_turns": ";".join(str(i) for i in flags["annotation_only_turns"]),
        "any_role_confusion": flags["any_role_confusion"],
        "any_annotation_only": bool(flags["annotation_only_turns"]),
        # Populated by sessions.iter_sessions; a crisis termination is a designed
        # outcome, not a defect, and is recorded so it can be excluded
        # deliberately rather than by accident.
        "session_status": meta.get("_status"),
        "session_status_reason": meta.get("_status_reason"),
    }


def flagged_turn_indices(qc: dict) -> set[int]:
    """The turns `qc_fields` marked, parsed back from the CSV-safe strings."""
    out: set[int] = set()
    for key in ("role_confused_turns", "annotation_only_turns"):
        value = qc.get(key) or ""
        out.update(int(part) for part in str(value).split(";") if part.strip())
    return out


def filter_turns(turns: list[tuple[int, str]], drop: set[int]) -> list[tuple[int, str]]:
    """QC view 2: keep the session, drop its flagged turns."""
    return [(i, t) for i, t in turns if i not in drop]
