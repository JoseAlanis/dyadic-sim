"""A case prior must not contain the vocabulary its detectors hunt for.

The patient agent is HANDED the prior and paraphrases it on every turn, so any
symptom term written into a case propagates into every transcript generated from
it - including the control sessions, where nothing was injected and the base rate
is supposed to be the thing you measure against.

One phrase is enough. "I've just felt a bit off for a while, not myself" seeds
`not myself`, a literal entry in `depressed_mood`'s `self_framed` list, into
every session of that case. The check is cheap, so it runs on every commit.

    uv run pytest tests/test_case_priors.py -q
"""

from __future__ import annotations

import glob
import os

import pytest
import yaml

from dyadic_sim.analysis.lexicon_detect import has_third_party_subject, is_negated
from dyadic_sim.analysis.symptom_lexicon import MADRS, PHQ9
from dyadic_sim.core import paths

# The user's config/ first, then the bundled defaults, as the generator resolves them.
CASE_DIRS = paths.config_dirs("priors", "patient", "cases")

#: Fields passed to the patient agent. Any other top-level key in a case file is
#: ignored by dyadic_sim/generation/priors/patient_prior.py, so it cannot leak vocabulary into speech
#: and is not linted.
#: Required in every case; the rest are optional and left out of the prompt
#: when absent (since prompt variant 7b).
REQUIRED = ("presenting_complaint",)
OPTIONAL = (
    "everyday_context",
    "theory_of_cure",
    "relational_pattern",
    "transference_expectation",
    "resistance_structure",
)
PATIENT_VISIBLE = REQUIRED + OPTIONAL

#: Cases whose whole purpose is to carry a theme that overlaps a symptom domain.
#: They are exempt, but the exemption is per-case and explicit: a bed is either
#: declared saturated or expected to be clean, never accidentally either.
#:
#: `feeling_off` is NOT exempt. It is listed as a known failure so the test
#: reports it as such rather than silently tolerating it; see `xfail` below.
SATURATED_BY_DESIGN = {
    "empty_and_invisible": "narcissistic wound; supplies depressive material on purpose",
    "only_love_can_save_me": "high frame-hazard bed; supplies relational and numbness material",
    "afraid_of_dogs": "anxiety bed; tension vocabulary is the case, not a leak",
}

KNOWN_CONTAMINATED = {
    "feeling_off": (
        "'not myself' is a depressed_mood term. Written as the neutral bed and "
        "floods numbness detectors in its own controls as a result."
    ),
}


def _case_files():
    seen = set()
    for directory in CASE_DIRS:
        for path in sorted(glob.glob(os.path.join(directory, "*.yaml"))):
            name = os.path.basename(path)[: -len(".yaml")]
            if name.startswith("_") or "_run_" in name or name in seen:
                continue          # template, generated per-condition cases, shadowed defaults
            seen.add(name)
            yield name, path


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _visible_text(path: str) -> str:
    case = _load(path)
    return " ".join(str(case.get(k) or "") for k in PATIENT_VISIBLE)


def detector_hits(text: str) -> dict[str, list[str]]:
    """Every symptom term a detector would ACCEPT in this text.

    Applies the same suppressions the real scan does, so a term that would be
    rejected as negated or third-party is not reported as a leak.
    """
    hits: dict[str, list[str]] = {}
    for panel in (PHQ9, MADRS):
        for symptom, lexicon in panel.items():
            found = [
                m.group(0)
                for m in lexicon.pattern().finditer(text)
                if not is_negated(text, m.start())
                and not has_third_party_subject(text, m.start())
            ]
            if found:
                hits.setdefault(symptom, []).extend(found)
    return hits


CASES = list(_case_files())
CLEAN_EXPECTED = [
    (n, p) for n, p in CASES if n not in SATURATED_BY_DESIGN and n not in KNOWN_CONTAMINATED
]


@pytest.mark.parametrize(("name", "path"), CLEAN_EXPECTED,
                         ids=[n for n, _ in CLEAN_EXPECTED])
def test_case_prior_contains_no_symptom_vocabulary(name, path):
    """A bed not declared saturated must not seed any detector's vocabulary."""
    hits = detector_hits(_visible_text(path))
    assert not hits, (
        f"case {name!r} seeds symptom vocabulary the detectors hunt for: {hits}. "
        f"The patient agent is given this text, so every session generated from "
        f"this case inherits it - including the controls, whose base rate is what "
        f"the injected arms are measured against. Either reword the prior or add "
        f"{name!r} to SATURATED_BY_DESIGN with a reason."
    )


@pytest.mark.parametrize("name", sorted(KNOWN_CONTAMINATED),
                         ids=sorted(KNOWN_CONTAMINATED))
@pytest.mark.xfail(strict=True, reason="documented contamination, kept visible")
def test_known_contaminated_cases_still_are(name):
    """Pins a known failure so it cannot be quietly 'fixed' or quietly forgotten.

    Marked `xfail(strict=True)`: if someone rewords the case so it comes out
    clean, this test FAILS to remind them to move it out of KNOWN_CONTAMINATED -
    and any result computed on the old wording is no longer comparable.
    """
    path = paths.config_path("priors", "patient", "cases", f"{name}.yaml")
    assert not detector_hits(_visible_text(path))


def test_every_case_declares_the_fields_the_patient_agent_is_given():
    """The required field is present; an optional one, if declared, is not empty;
    and no key is unknown, so a misspelt field cannot silently drop a section."""
    for name, path in CASES:
        case = _load(path)
        missing = [k for k in REQUIRED if not str(case.get(k) or "").strip()]
        empty = [k for k in OPTIONAL if k in case and not str(case[k] or "").strip()]
        unknown = sorted(set(case) - set(PATIENT_VISIBLE))
        assert not missing, f"case {name!r} is missing required fields: {missing}"
        assert not empty, f"case {name!r} declares empty fields: {empty}"
        assert not unknown, f"case {name!r} has unknown fields: {unknown}"
