"""Golden hashes of every system prompt the generator can render, and the opening.

Moving or reorganising code must not change a byte of what the agents are
told. Each hand-written case is
rendered as the batch runner renders it: without symptoms, and with each of the
nine PHQ-9 symptoms at "nearly every day" under both injection schemes. Every
therapist variant is rendered too. Their SHA-256 are pinned in
`tests/golden/prompts.json`, captured at generation_version 6.

A deliberate change to a prior or the opening regenerates the file in the same
commit, and the commit says why:

    DYADIC_SIM_UPDATE_GOLDEN=1 uv run pytest tests/test_golden_prompts.py -q
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from dyadic_sim.cli.run_symptom_experiments import (
    build_no_symptoms_case,
    build_single_symptom_case,
)
from dyadic_sim.core.vocabulary import KEYS
from dyadic_sim.generation.opening import FIRST_SESSION_OPENING
from dyadic_sim.generation.priors import patient_prior
from dyadic_sim.generation.priors.loader import load_patient_case
from dyadic_sim.generation.priors.therapist_prior import build_therapist_prior

GOLDEN = Path(__file__).parent / "golden" / "prompts.json"
PRIORS = Path(__file__).resolve().parents[1] / "config" / "priors"
SCHEMES = ("all_nine", "active_only")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hand_written(directory: Path) -> list[str]:
    """Case or variant names, without templates and generated per-condition files."""
    return sorted(
        p.stem
        for p in directory.glob("*.yaml")
        if not p.stem.startswith("_") and "_run_" not in p.stem
    )


def _render_all(monkeypatch) -> dict:
    rendered: dict[str, dict[str, str]] = {"opening": {}, "patient": {}, "therapist": {}}
    rendered["opening"]["first_session"] = _sha(FIRST_SESSION_OPENING)

    for case in _hand_written(PRIORS / "patient" / "cases"):
        base = load_patient_case(case)
        conditions = {"no_symptoms": build_no_symptoms_case(base)}
        for key in KEYS:
            for scheme in SCHEMES:
                conditions[f"{key}/{scheme}"] = build_single_symptom_case(
                    base, key, injection=scheme
                )
        for condition, case_data in conditions.items():
            monkeypatch.setattr(patient_prior, "load_patient_case", lambda _, d=case_data: d)
            prompt = patient_prior.build_patient_prior(case).build_system_prompt()
            rendered["patient"][f"{case}/{condition}"] = _sha(prompt)

    for variant in _hand_written(PRIORS / "therapist" / "variants"):
        prompt = build_therapist_prior(variant).build_system_prompt()
        rendered["therapist"][variant] = _sha(prompt)
    return rendered


def test_rendered_prompts_match_the_golden_hashes(monkeypatch):
    rendered = _render_all(monkeypatch)
    if os.getenv("DYADIC_SIM_UPDATE_GOLDEN"):
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(rendered, indent=2, sort_keys=True) + "\n")

    golden = json.loads(GOLDEN.read_text())
    changed = sorted(
        f"{group}/{name}"
        for group in golden.keys() | rendered.keys()
        for name in golden.get(group, {}).keys() | rendered.get(group, {}).keys()
        if golden.get(group, {}).get(name) != rendered.get(group, {}).get(name)
    )
    assert not changed, f"{len(changed)} prompt(s) differ from tests/golden/prompts.json: {changed}"
