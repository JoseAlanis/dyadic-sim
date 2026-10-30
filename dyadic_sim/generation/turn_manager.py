"""
dyadic_sim/generation/turn_manager.py

Builds the full context passed to each agent on every turn:
    system_prompt = the agent's prior
    messages      = the conversation so far, in the order it was spoken

The conversation starts with the therapist's opening, saved with the session as
`initial_patient_prompt`, and alternates from there: opening, P1, T1, P2, T2, ...
Each agent sees its own lines as "assistant" and the other's as "user", so the
patient's list starts with the opening as a user message and the therapist's
with the opening as its own first line.

The context window is the agent's only view of the world. There is no other
carrier of memory: nothing is summarised or injected between turns, so the
prior is the only constant and the transcript the only record.
"""

from dyadic_sim.generation.agents.base_agent import Message
from dyadic_sim.generation.priors.patient_prior import PatientPrior
from dyadic_sim.generation.priors.therapist_prior import TherapistPrior

# How many previous turns to include in the message history. None means the
# whole session. Nothing summarises what falls outside a finite window, so a
# window silently forgets; set one only when a session would not fit the
# model's context.
HISTORY_WINDOW: int | None = None

PATIENT = "patient"
THERAPIST = "therapist"


def build_therapist_context(
    prior: TherapistPrior,
    opening: str,
    history: list[tuple[str, str]],   # list of (patient_text, therapist_text)
    patient_turn: str,
    turn: int | None = None,
) -> tuple[str, list[Message]]:
    """
    Build system prompt + message list for a therapist turn.

    Args:
        prior:         The therapist's prior object
        opening:       The therapist's opening line, as saved with the session
        history:       Completed (patient, therapist) turn pairs
        patient_turn:  What the patient just said, which the therapist answers
        turn:          The reply being written (k answers the patient's k-th
                       turn); selects the intake agenda step

    Returns:
        (system_prompt, messages) ready to pass to agent.complete()
    """
    utterances = _conversation(opening, history) + [(PATIENT, patient_turn)]
    return prior.build_system_prompt(turn), _as_messages(utterances, own=THERAPIST)


def build_patient_context(
    prior: PatientPrior,
    opening: str,
    history: list[tuple[str, str]],   # list of (patient_text, therapist_text)
) -> tuple[str, list[Message]]:
    """
    Build system prompt + message list for a patient turn.

    Args:
        prior:    The patient's prior object
        opening:  The therapist's opening line, as saved with the session
        history:  Completed (patient, therapist) turn pairs; the patient answers
                  the last therapist line, or the opening on the first turn

    Returns:
        (system_prompt, messages) ready to pass to agent.complete()
    """
    utterances = _conversation(opening, history)
    return prior.build_system_prompt(), _as_messages(utterances, own=PATIENT)


def _conversation(opening: str, history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """(speaker, text) in the order spoken, windowed by HISTORY_WINDOW.

    A transcript row holds a patient line and the therapist's reply to it; the
    opening precedes the first row. A window drops whole rows from the start,
    and the opening with them once any row is dropped.
    """
    kept = history if HISTORY_WINDOW is None else history[-HISTORY_WINDOW:]
    utterances = [] if len(kept) < len(history) else [(THERAPIST, opening)]
    for patient_text, therapist_text in kept:
        utterances += [(PATIENT, patient_text), (THERAPIST, therapist_text)]
    return utterances


def _as_messages(utterances: list[tuple[str, str]], own: str) -> list[Message]:
    """One message per utterance: the agent's own lines as assistant, the other's as user.

    An empty line (a turn that was only stage directions, emptied by
    clean_utterance) is left out of the history, as before version 6. The line
    being answered is always sent.
    """
    last = len(utterances) - 1
    return [
        Message(role="assistant" if speaker == own else "user", content=text)
        for i, (speaker, text) in enumerate(utterances)
        if text or i == last
    ]
