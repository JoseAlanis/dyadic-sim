"""What generation_version 7, prompt variant 7a, tells each agent, and how it is enforced.

The golden hashes pin every rendered prompt byte for byte; these tests state
the decisions behind them, so a later variant changes them on purpose.

    uv run pytest tests/test_prompts_v7.py -q
"""

from __future__ import annotations

import json

import pytest

import dyadic_sim.generation.dyad as dyad_mod
import dyadic_sim.generation.session as session_mod
from dyadic_sim.cli.run_symptom_experiments import (
    build_no_symptoms_case,
    build_single_symptom_case,
)
from dyadic_sim.generation.agents.base_agent import AgentResponse, BaseAgent
from dyadic_sim.generation.dyad import Dyad
from dyadic_sim.generation.priors import patient_prior
from dyadic_sim.generation.priors.loader import load_patient_case
from dyadic_sim.generation.priors.therapist_prior import build_therapist_prior
from dyadic_sim.generation.session import IncompatibleSessionError
from dyadic_sim.generation.utterance import (
    GENERATION_VERSION,
    PROMPT_VARIANT,
    STAY_WITH_WHAT_IS_SAID,
)

CASE = "no_complaint"
AGENDA = ["follow_cue", "why_now", "mood_thoughts", "sleep_appetite", "relationships",
          "daily_functioning", "strengths", "goals", "summary", "closing"]


def _patient_prompt(monkeypatch, case_data) -> str:
    monkeypatch.setattr(patient_prior, "load_patient_case", lambda _: case_data)
    return patient_prior.build_patient_prior(CASE).build_system_prompt()


def _health_section(prompt: str) -> str:
    start = prompt.index("## Your Health Lately")
    return prompt[start:prompt.index("\n---\n", start)]


# --- Patient ------------------------------------------------------------------


def test_the_patient_frame_presumes_no_problem(monkeypatch):
    prompt = _patient_prompt(monkeypatch, load_patient_case(CASE))

    assert "first appointment" in prompt
    assert "something is wrong" not in prompt
    assert "you will not find much" not in prompt  # no instruction to deny
    assert "## Your Health Lately" not in prompt  # nothing declared, nothing rendered


def test_the_sleep_arm_states_the_facts_and_when_to_disclose_them(monkeypatch):
    case = build_single_symptom_case(load_patient_case(CASE), "sleep_problems",
                                     "nearly every day", injection="active_only")
    section = _health_section(_patient_prompt(monkeypatch, case))

    assert "difficulty falling asleep nearly every night" in section
    assert "for about a month" in section
    assert "managing at work" in section
    assert "can't say what causes it" in section
    assert "haven't tried anything in particular" in section
    assert "You don't bring this up yourself; when the therapist asks about your sleep" in section
    assert "tired" not in section  # no added feature for the low-energy detector
    assert "Do not invent problems" in section


def test_the_control_declares_the_compared_symptom_absent(monkeypatch):
    case = build_no_symptoms_case(load_patient_case(CASE), absent=["sleep_problems"])
    section = _health_section(_patient_prompt(monkeypatch, case))

    assert "Sleep: you sleep well" in section
    assert "difficulty" not in section
    assert "Do not invent problems" in section


def test_all_nine_names_the_eight_others_as_absent(monkeypatch):
    case = build_single_symptom_case(load_patient_case(CASE), "sleep_problems",
                                     "nearly every day", injection="all_nine")
    section = _health_section(_patient_prompt(monkeypatch, case))

    assert section.count("not a problem for you") == 8
    assert "difficulty falling asleep" in section


def test_the_patient_is_asked_for_plain_speech(monkeypatch):
    prompt = _patient_prompt(monkeypatch, load_patient_case(CASE))

    assert "everyday words" in prompt
    assert "Don't ask rhetorical questions" in prompt
    assert "don't repeat what you have already said" in prompt
    assert "let the words carry it" not in prompt


# --- Therapist ----------------------------------------------------------------


def test_the_therapist_runs_an_intake_without_treatment_techniques():
    prompt = build_therapist_prior("cbt").build_system_prompt()

    assert "first appointment, an intake" in prompt
    assert "No Socratic questioning" in prompt
    assert "Socratic questioning provides guidance" not in prompt  # v6 technique
    assert "review homework" not in prompt
    assert "Follow what they bring" not in STAY_WITH_WHAT_IS_SAID
    assert "never suggest that they have a problem" in STAY_WITH_WHAT_IS_SAID


def test_each_reply_gets_its_agenda_step():
    prior = build_therapist_prior("cbt")

    assert [step["id"] for step in prior.intake_agenda] == AGENDA
    for turn in range(1, 11):
        assert f"This is your reply {turn} of 10" in prior.build_system_prompt(turn)
    assert "How have you been sleeping and eating lately?" in prior.build_system_prompt(4)
    assert "whether anything is missing" in prior.build_system_prompt(9)
    closing = prior.build_system_prompt(10).split("## This Reply")[1]
    assert "Do not ask a question" in closing
    assert "## This Reply" not in prior.build_system_prompt(11)
    assert "risk" not in json.dumps(prior.intake_agenda).lower()


# --- Identity and enforcement ------------------------------------------------


class StubAgent(BaseAgent):
    def complete(self, system_prompt, messages, max_tokens=512, temperature=0.7):
        return AgentResponse(content=f"{self.role} {len(messages)}", model=self.model)

    def is_available(self) -> bool:
        return True


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(dyad_mod, "build_agent", lambda model, role: StubAgent(model, role))
    return tmp_path


def test_sessions_record_the_version_variant_and_agenda_step(sessions_dir):
    session = Dyad("stub", "stub", CASE, orientation="cbt").run(n_turns=10)

    meta = json.loads((session.session_dir / "metadata.json").read_text())
    rows = [json.loads(line) for line in (session.session_dir / "transcript.jsonl").read_text().splitlines()]
    assert (meta["generation_version"], meta["prompt_variant"]) == (GENERATION_VERSION, PROMPT_VARIANT)
    assert [row["therapist_diagnostics"]["agenda_step"] for row in rows] == AGENDA
    assert all("agenda_step" not in row["patient_diagnostics"] for row in rows)


@pytest.mark.parametrize("variant", ["7z", None], ids=["other-variant", "no-variant"])
def test_a_session_of_another_variant_is_not_continued(sessions_dir, variant):
    session = Dyad("stub", "stub", CASE, orientation="cbt").run(n_turns=1)
    path = session.session_dir / "metadata.json"
    meta = json.loads(path.read_text())
    if variant is None:
        del meta["prompt_variant"]
    else:
        meta["prompt_variant"] = variant
    path.write_text(json.dumps(meta))

    with pytest.raises(IncompatibleSessionError, match="prompt variant"):
        Dyad("stub", "stub", CASE, orientation="cbt", session_id=session.session_id)
