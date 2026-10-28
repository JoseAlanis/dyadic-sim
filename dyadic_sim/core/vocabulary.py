"""The prior vocabulary: the PHQ-9 taxonomy patient priors are WRITTEN IN.

Named for its role rather than for the instrument, because "instrument" cannot
distinguish the two taxonomies in this project - PHQ-9 and MADRS are both
clinical instruments, and the question that keeps arising is which belongs
where. The distinction is what each one is FOR:

    prior_vocabulary.py       PHQ-9   what priors are written in   (injected)
    instruments/madrs.py      MADRS   what the rater scores against (measured)

`PHQ_TO_TOPIC` in `madrs.py` is the bridge between them, and it points from
these keys to MADRS topics, which is the direction that tells you which side is
which.

Same nine items, three registers, none interchangeable.

This module exists because the three were previously scattered - the labels in
`dyadic_sim/generation/priors/patient_prior.py` and `dyadic_sim/scoring/config.py`, the reference
sentences inside the `symptom_embed.py` CLI script - and were repeatedly
mistaken for copies of each other. They are not copies. They are different
strings for the same nine keys, and which one you want depends on the question:

  KEYS        canonical order. `target_z` indexes a similarity row by position,
              so the order is load-bearing, not cosmetic.

  LABELS      the official PHQ-9 item wording, third person. Rendered VERBATIM
              into the patient prompt by `dyadic_sim.generation.priors.patient_prior`, so these are
              literally "what the patient saw". Changing one changes the
              stimulus and invalidates comparison with existing runs.

  REFERENCES  hand-written first-person sentences, used only as embedding
              targets by the analysis. These are our wording, not the
              instrument's, which is exactly why
              `dyadic_sim/analysis/symptom_embed_robust.py` exists: it re-runs
              the measure against LABELS and against LABELS + the frequency
              clause to test how much the result depends on this wording.

Because of that last point LABELS and REFERENCES must stay distinct objects. If
they were unified, the robustness check would correlate a variable with itself
and report perfect agreement no matter what.

MADRS keeps its equivalents in `dyadic_sim/scoring/instruments/madrs.py`
(TOPICS, TOPIC_DESCRIPTIONS_EN); this is the counterpart on the prior side.

**Why this is not in `instruments/`.** That package is a registry: every module
in it defines an `Instrument` and registers it, which requires topics,
relevance patterns, a score range and a language, because an `Instrument` is a
taxonomy a rater scores a transcript against. Nothing rates a transcript
against PHQ-9 here. Moving this module there would make it the only member
absent from `REGISTRY`, so `get_instrument("phq9")` would raise - a trap rather
than a tidier layout.

Note the rater itself is not in `instruments/` either: that is
`madrs_bert_model.MadrsBertRegressor`, driven by `turn_scorer.TurnScorer`.
`instruments/` holds what the rater scores *against*.

This changes the day someone scores transcripts against PHQ-9 items directly.
Then a `PHQ9 = Instrument(...)` belongs in `instruments/` and should consume the
vocabulary defined here rather than restate it.
"""

from __future__ import annotations

# Third person, official wording. Goes into the prompt unchanged.
LABELS = {
    'lack_of_pleasure': 'Little interest or pleasure in doing things',
    'depressed_mood': 'Feeling down, depressed, or hopeless',
    'sleep_problems': 'Sleep problems',
    'low_energy': 'Feeling tired or having little energy',
    'appetite_changes': 'Poor appetite or overeating',
    'feelings_of_failure_or_guilt': 'Feeling bad about yourself, guilty, or like a failure',
    'concentration_problems': 'Trouble concentrating',
    'psychomotor_changes': 'Moving or speaking slowly, or feeling restless',
    'thoughts_of_death_or_self_harm': 'Thoughts that you would be better off dead or of hurting yourself',
}


# First person, our phrasing. Embedding targets for the analysis only;
# never shown to any agent.
REFERENCES = {
    'lack_of_pleasure': (
        'I have little interest or pleasure in doing things; nothing feels enjoyable anymore.'
    ),
    'depressed_mood': 'I feel down, depressed, sad, and hopeless most of the time.',
    'sleep_problems': (
        "I have trouble sleeping; I can't fall or stay asleep, or I sleep too much."
    ),
    'low_energy': (
        'I feel tired and have very little energy; I am exhausted and worn out.'
    ),
    'appetite_changes': (
        'My appetite has changed; I have a poor appetite or I am overeating.'
    ),
    'feelings_of_failure_or_guilt': 'I feel bad about myself, like a failure, worthless, and guilty.',
    'concentration_problems': (
        "I have trouble concentrating and focusing; my mind won't stay on things."
    ),
    'psychomotor_changes': (
        "I move or speak slowly, or I feel so restless and fidgety I can't sit still."
    ),
    'thoughts_of_death_or_self_harm': (
        'I have thoughts that I would be better off dead or of hurting myself.'
    ),
}


# Canonical order. Both dicts must agree with it; tests/test_prior_vocabulary.py pins that.
KEYS = tuple(LABELS)
