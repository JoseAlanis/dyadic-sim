"""One PHQ-9 vocabulary, imported everywhere, pinned here.

`dyadic_sim/core/vocabulary.py` is the single home of the PHQ-9 keys and
their three registers: KEYS, LABELS (prompt wording) and REFERENCES (analysis
probes). The prior, the scoring config, the batch runner and
`symptom_embed_robust.py` all import it. LABELS and REFERENCES are deliberately
not unified: `symptom_embed_robust.py` checks whether the measure in
`symptom_embed.py` survives a change of reference wording, so merging them would
make it correlate a variable with itself, and a local copy of the references
would let it compare new wording against stale wording and still agree.

These tests fail if anyone reintroduces a local copy, or unifies the two
registers that must stay distinct.

    uv run pytest tests/ -q
"""

from __future__ import annotations

import inspect

from dyadic_sim.analysis import symptom_embed_robust as robust
from dyadic_sim.analysis.symptom_embed import REFERENCES
from dyadic_sim.analysis.symptom_lexicon import PHQ9
from dyadic_sim.cli.run_symptom_experiments import SYMPTOM_KEYS
from dyadic_sim.core.vocabulary import KEYS, LABELS
from dyadic_sim.generation.priors import patient_prior
from dyadic_sim.scoring.config import PHQ_SYMPTOM_LABELS
from dyadic_sim.scoring.instruments.madrs import PHQ_MAPPING_QUALITY, PHQ_TO_TOPIC

CANONICAL = list(PHQ_SYMPTOM_LABELS)


def test_there_are_nine_items():
    assert len(CANONICAL) == 9


def test_keys_is_the_canonical_order():
    """KEYS is the tuple every consumer derives its order from; pin it."""
    assert tuple(CANONICAL) == KEYS
    assert list(KEYS) == SYMPTOM_KEYS


def test_every_consumer_uses_the_same_keys_in_the_same_order():
    """Order matters: `target_z` indexes the similarity row by position."""
    assert SYMPTOM_KEYS == CANONICAL
    assert list(REFERENCES) == CANONICAL
    assert robust.SYMPTOMS == CANONICAL
    assert list(PHQ9) == CANONICAL


def test_reference_sets_are_the_originals_not_copies():
    """The robustness check must derive its sets, never restate them.

    Identity (`is`) used to carry this, but the sets are now tuples-per-symptom
    so that every reference set has one shape. Content equality plus the
    source-level check below is a stronger guard anyway: identity would still
    pass if someone wrote `REF_A = dict(REFERENCES)`, whereas a pasted literal
    fails both.
    """
    for key in CANONICAL:
        assert robust.REFSETS["A_handwritten"][key] == (REFERENCES[key],), key
        assert robust.REFSETS["B_phq9_label"][key] == (PHQ_SYMPTOM_LABELS[key],), key


def test_robustness_check_contains_no_pasted_reference_text():
    """The failure this whole file exists to prevent.

    `symptom_embed_robust.py` tests whether the measure survives a change of
    reference wording. If it held its own copy of the sentences and that copy
    drifted, it would compare the new wording against a stale one and report
    high agreement regardless - a validity check that silently always passes.
    """
    source = inspect.getsource(robust)
    for key in CANONICAL:
        assert REFERENCES[key] not in source, (
            f"reference sentence for {key} is pasted into symptom_embed_robust.py; "
            f"import it instead"
        )
        assert PHQ_SYMPTOM_LABELS[key] not in source, (
            f"prompt label for {key} is pasted into symptom_embed_robust.py; "
            f"import it instead"
        )


def test_madrs_maps_from_exactly_these_keys():
    """MADRS is the rating side and maps *from* the PHQ-9 prior side.

    `PHQ_TO_TOPIC` and `PHQ_MAPPING_QUALITY` are genuinely different content
    from anything in prior_vocabulary.py (they are mappings, not wordings), so they cannot
    be derived from KEYS. Their key sets can still be pinned: a PHQ-9 item that
    exists here but is missing from the map would silently drop out of the
    MADRS panel, which is how `psychomotor_changes` is legitimately absent
    (it maps to None) and how an accidental omission would look identical.
    """
    assert tuple(PHQ_TO_TOPIC) == tuple(CANONICAL)
    assert tuple(PHQ_MAPPING_QUALITY) == tuple(CANONICAL)


def test_labels_and_references_stay_distinct():
    """They describe the same nine items but are not interchangeable.

    LABELS is the instrument's own third-person wording and goes into the
    prompt; REFERENCES is our first-person phrasing and is only ever an
    embedding target. `symptom_embed_robust` measures how much the result
    depends on that difference, so unifying them would make its comparison
    vacuous.
    """
    assert LABELS is not REFERENCES
    for key in CANONICAL:
        assert LABELS[key] != REFERENCES[key], key


def test_ref_c_extends_the_real_prompt_labels():
    """Set C is the full prompt line: the label plus the frequency clause."""
    for key, label in PHQ_SYMPTOM_LABELS.items():
        (text,) = robust.REFSETS["C_phq9_full_line"][key]
        assert text.startswith(label), key
        assert len(text) > len(label), f"{key}: C adds nothing to the label"


def test_patient_prior_renders_the_canonical_labels():
    """The labels are 'what the patient saw', so the prompt must use these."""
    source = inspect.getsource(patient_prior)
    assert "PHQ_SYMPTOM_LABELS.get(" in source, (
        "patient_prior must import the labels, not redeclare them"
    )
    for label in PHQ_SYMPTOM_LABELS.values():
        assert label not in source, f"local copy of a prompt label reintroduced: {label!r}"
