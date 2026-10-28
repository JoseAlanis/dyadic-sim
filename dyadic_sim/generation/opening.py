"""How a session opens: the therapist's first words.

The opening is the therapist's first utterance. It is saved with the session as
`initial_patient_prompt` when the session starts, and both agents' histories
begin with it (see dyadic_sim/generation/turn_manager.py), so the patient's first turn is a
reply to it and a resumed session keeps the opening it started with.

It is speech, not an instruction to the patient. An instruction to state the
presenting complaint forces a recited chief complaint into turn 1 and makes
first turns alike across sessions, where the design wants symptoms to surface as
lived texture the patient raises themselves.

Two constraints on the wording, both deliberate:

* **Never name a symptom domain.** The opening is the cheapest available version
  of a semi-structured intake, so a broad invitation gets some of the benefit for
  free. But "how have you been sleeping?" would leak the manipulation outright.
  Invite the patient to speak broadly; let them choose what is salient.
* **Explain the prior contact.** Referring to the booking call is why the
  therapist can ask for it "in your own words" without implausibly knowing
  nothing, and it licenses a discursive answer rather than a prepared statement.
"""

from __future__ import annotations

# The opening used for generation.
FIRST_SESSION_OPENING = (
    "Hi, please, come on in; good to meet you properly. We spoke briefly when you booked "
    "the appointment, but I'd rather hear it in your own words: what's been "
    "going on for you?"
)

# The opening of records whose metadata has no `initial_patient_prompt`
# (generation_version 1 and 2). Not for generation:
# `dyadic_sim/scoring/transcript_parser.py` uses it to reconstruct the first
# therapist turn of those records, so changing it would re-pair them and shift
# their scores.
V1_OPENING = (
    "You have just arrived for a therapy session. "
    "The therapist is present and waiting. Say what brings you here."
)


def opening_utterance(
    session_number: int = 1,
    previous_summary: str | None = None,
) -> str:
    """The therapist's opening line for a session with this patient.

    Args:
        session_number:   1-indexed. Only session 1 is implemented.
        previous_summary: What the previous session covered. Unused for now;
                          present so the session 2+ branch has the input it
                          needs without reshaping callers.

    A longitudinal corpus would extend this with a resumption opening ("last
    time you mentioned…"). That branch has a design decision attached: generating
    it from the therapist's state reads naturally but makes the opening vary per
    session, so it stops being a controlled constant. Single-session work does
    not hit that, which is why only session 1 exists here.
    """
    if session_number == 1:
        return FIRST_SESSION_OPENING
    raise NotImplementedError(
        "Only session 1 is implemented. A session-2+ opening needs "
        "previous_summary and a decision on generated vs templated text. "
        "See the docstring."
    )
