"""The draft codebook shown to raters, as data.

Protocol `draft-0`: the wording is a starting point for calibration, not a
frozen instrument. Machine keys are the nine in
`dyadic_sim.core.vocabulary.KEYS`; clinician-facing labels are kept here,
separately, so that rewording a label can never change a key.
"""

from __future__ import annotations

from dyadic_sim.core.vocabulary import KEYS

PROTOCOL_VERSION = "draft-0"

#: (key, short label, PHQ-9 item wording). Item wording from the PHQ-9, which is
#: free to use. Shown so raters know what each key covers; the frequency window
#: is stated once in the session form, not repeated here.
SYMPTOMS = [
    ("lack_of_pleasure", "Loss of interest or pleasure",
     "Little interest or pleasure in doing things"),
    ("depressed_mood", "Low mood",
     "Feeling down, depressed, or hopeless"),
    ("sleep_problems", "Sleep",
     "Trouble falling or staying asleep, or sleeping too much"),
    ("low_energy", "Tiredness or low energy",
     "Feeling tired or having little energy"),
    ("appetite_changes", "Appetite",
     "Poor appetite or overeating"),
    ("feelings_of_failure_or_guilt", "Guilt, worthlessness or failure",
     ("Feeling bad about yourself, or that you are a failure or have let yourself "
      "or your family down")),
    ("concentration_problems", "Concentration",
     "Trouble concentrating on things, such as reading or watching television"),
    ("psychomotor_changes", "Slowed down or restless",
     ("Moving or speaking so slowly that others could notice, or being so fidgety "
      "or restless that you move around much more than usual")),
    ("thoughts_of_death_or_self_harm", "Thoughts of death or self-harm",
     "Thoughts that you would be better off dead, or of hurting yourself in some way"),
]

CLAIM_CODES = [
    ("present", "Present",
     "The patient reports this as their own current or recent experience."),
    ("raised_and_denied", "Raised and denied",
     ("The symptom comes up and the patient says it does not apply to them, "
      "for example “I'm sleeping fine”.")),
    ("unclear", "Unclear",
     ("Relevant but not a current report: past or out-of-window experience, "
      "someone else's experience, a hypothetical, quoted speech, unclear timing, "
      "or you cannot tell.")),
]

ELICITED = [
    ("yes", "Yes", "Answers a therapist question specifically about this symptom."),
    ("no", "No", ("Volunteered, or in answer to a broad question such as "
                  "“how have things been?”.")),
    ("unclear", "Unclear", "You cannot tell whether the question was about this symptom."),
]

EXPRESSION = [
    ("present", "Present", "Reported as current or recent experience somewhere in the session."),
    ("raised_and_denied", "Raised and denied", "Came up, and the patient said it does not apply."),
    ("unclear", "Unclear", "Relevant material, but no clear current report or denial."),
    ("no_evidence", "No evidence", "Nothing in the session bears on this symptom."),
]

FREQUENCY = [
    ("0", "0 · Not at all", ""),
    ("1", "1 · Several days", ""),
    ("2", "2 · More than half the days", ""),
    ("3", "3 · Nearly every day", ""),
    ("cannot_judge", "Cannot judge", "The session gives no basis for a frequency."),
]

PROMINENCE_EXTRA = [
    ("none", "None of the nine", ""),
    ("cannot_determine", "Cannot determine", ""),
]

REALISM = [
    (1, "1 · Not at all like a real patient"),
    (2, "2"),
    (3, "3"),
    (4, "4"),
    (5, "5 · Entirely like a real patient"),
]

if [key for key, _, _ in SYMPTOMS] != list(KEYS):
    raise RuntimeError("codebook symptoms are out of step with prior_vocabulary.KEYS")


def keys(options) -> set:
    """The machine values of one option list."""
    return {option[0] for option in options}


def as_payload() -> dict:
    """The codebook as the page receives it."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "symptoms": [{"key": k, "label": label, "item": item} for k, label, item in SYMPTOMS],
        "claim_codes": [{"key": k, "label": label, "help": h} for k, label, h in CLAIM_CODES],
        "elicited": [{"key": k, "label": label, "help": h} for k, label, h in ELICITED],
        "expression": [{"key": k, "label": label, "help": h} for k, label, h in EXPRESSION],
        "frequency": [{"key": k, "label": label, "help": h} for k, label, h in FREQUENCY],
        "prominence_extra": [{"key": k, "label": label} for k, label, _ in PROMINENCE_EXTRA],
        "realism": [{"key": k, "label": label} for k, label in REALISM],
    }
