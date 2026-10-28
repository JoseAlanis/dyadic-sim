"""What each agent is sent, turn by turn, and when a saved session may continue.

These tests pin the complete requests at the agent boundary, with stub agents
in place of a model server: both agents receive the conversation in the order
it was spoken, starting from the saved opening. They also pin the rule that
only a session started by the current generator can be continued.

    uv run pytest tests/test_conversation_history.py -q
"""

from __future__ import annotations

import json
from itertools import pairwise

import pytest

import dyadic_sim.generation.dyad as dyad_mod
import dyadic_sim.generation.session as session_mod
from dyadic_sim.generation.agents.base_agent import AgentResponse, BaseAgent
from dyadic_sim.generation.dyad import Dyad
from dyadic_sim.generation.opening import FIRST_SESSION_OPENING
from dyadic_sim.generation.priors.patient_prior import PatientPrior
from dyadic_sim.generation.session import resume_session
from dyadic_sim.generation.utterance import GENERATION_VERSION

CASE = "no_complaint"
OPENING = FIRST_SESSION_OPENING

# The complete message lists for three turns. The stub answers "<role> <number
# of messages it was sent>", so P1 = "patient 1", T1 = "therapist 2", and so on.
EXPECTED_PATIENT = [
    [("user", OPENING)],
    [("user", OPENING), ("assistant", "patient 1"), ("user", "therapist 2")],
    [
        ("user", OPENING),
        ("assistant", "patient 1"),
        ("user", "therapist 2"),
        ("assistant", "patient 3"),
        ("user", "therapist 4"),
    ],
]
EXPECTED_THERAPIST = [
    [("assistant", OPENING), ("user", "patient 1")],
    [
        ("assistant", OPENING),
        ("user", "patient 1"),
        ("assistant", "therapist 2"),
        ("user", "patient 3"),
    ],
    [
        ("assistant", OPENING),
        ("user", "patient 1"),
        ("assistant", "therapist 2"),
        ("user", "patient 3"),
        ("assistant", "therapist 4"),
        ("user", "patient 5"),
    ],
]


class RecordingAgent(BaseAgent):
    """Records every request and answers from the request alone."""

    def __init__(self, model: str, role: str):
        super().__init__(model, role)
        self.requests: list[tuple[str, list[tuple[str, str]]]] = []

    def complete(self, system_prompt, messages, max_tokens=512, temperature=0.7):
        self.requests.append((system_prompt, [(m.role, m.content) for m in messages]))
        # A function of the request, so a resumed run answers exactly as an
        # uninterrupted one would.
        return AgentResponse(content=f"{self.role} {len(messages)}", model=self.model)

    def is_available(self) -> bool:
        return True


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def agents(monkeypatch):
    """Every agent a dyad builds, in order. No provider is ever contacted."""
    built: list[RecordingAgent] = []

    def build(model, role):
        agent = RecordingAgent(model, role)
        built.append(agent)
        return agent

    monkeypatch.setattr(dyad_mod, "build_agent", build)
    return built


def _latest(built, role):
    return [agent for agent in built if agent.role == role][-1]


def _dyad(**kwargs) -> Dyad:
    return Dyad("stub", "stub", CASE, orientation="cbt", **kwargs)


def _write_record(directory, version, opening=OPENING):
    """A one-turn session as an earlier generator would have left it."""
    directory.mkdir()
    meta = {
        "metadata_schema_version": 2,
        "session_id": directory.name,
        "patient_id": None,
        "therapist_model": "stub",
        "patient_model": "stub",
        "case_name": CASE,
        "orientation": "cbt",
        "patient_symptoms": "",
        "patient_symptom_levels": {},
        "conversation_language": "en",
        "initial_patient_prompt": opening,
        "turn_count": 1,
        "created_at": "2026-09-08T10:00:00",
    }
    if version is not None:  # absent means version 1
        meta["generation_version"] = version
    (directory / "metadata.json").write_text(json.dumps(meta))
    row = {
        "turn": 1,
        "therapist_text": "What brings you here?",
        "patient_text": "The garage, mostly.",
        "therapist_model": "stub",
        "patient_model": "stub",
        "therapist_tokens": None,
        "patient_tokens": None,
    }
    (directory / "transcript.jsonl").write_text(json.dumps(row) + "\n")
    return directory


def _snapshot(directory):
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(directory.iterdir())}


def test_both_agents_read_the_conversation_in_order_from_the_opening(sessions_dir, agents):
    dyad = _dyad()
    dyad.run(n_turns=3)

    patient, therapist = _latest(agents, "patient"), _latest(agents, "therapist")
    assert [messages for _, messages in patient.requests] == EXPECTED_PATIENT
    assert [messages for _, messages in therapist.requests] == EXPECTED_THERAPIST
    # The system prompt is the prior alone, the same on every turn.
    assert {system for system, _ in patient.requests} == {
        dyad.patient_prior.build_system_prompt()
    }
    assert {system for system, _ in therapist.requests} == {
        dyad.therapist_prior.build_system_prompt()
    }


def test_every_recorded_utterance_is_sent_once_in_the_order_spoken(sessions_dir, agents):
    session = _dyad().run(n_turns=3)

    spoken = [session.initial_patient_prompt]
    for record in session.transcript:
        spoken += [record.patient_text, record.therapist_text]

    patient, therapist = _latest(agents, "patient"), _latest(agents, "therapist")
    # Before its third line the patient has heard everything up to T2; the
    # therapist answers P3 with everything before it in view.
    assert [text for _, text in patient.requests[-1][1]] == spoken[:5]
    assert [text for _, text in therapist.requests[-1][1]] == spoken[:6]
    for _, messages in patient.requests + therapist.requests:
        roles = [role for role, _ in messages]
        assert all(a != b for a, b in pairwise(roles))


def test_a_resumed_session_uses_its_saved_opening(sessions_dir, agents):
    session = _dyad().run(n_turns=1)
    meta_path = session.session_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["initial_patient_prompt"] = "Saved opening."
    meta_path.write_text(json.dumps(meta))

    _dyad(session_id=session.session_id).run(n_turns=1)

    assert _latest(agents, "patient").requests[0][1][0] == ("user", "Saved opening.")
    assert _latest(agents, "therapist").requests[0][1][0] == ("assistant", "Saved opening.")


def test_resuming_sends_the_same_next_requests_as_an_uninterrupted_run(sessions_dir, agents):
    straight = _dyad().run(n_turns=3)
    expected = {role: _latest(agents, role).requests[2] for role in ("patient", "therapist")}

    first = _dyad().run(n_turns=2)
    _dyad(session_id=first.session_id).run(n_turns=1)
    resumed = {role: _latest(agents, role).requests[0] for role in ("patient", "therapist")}

    assert resumed == expected
    lines = [(r.patient_text, r.therapist_text) for r in straight.transcript]
    reloaded = resume_session(first.session_id)
    assert [(r.patient_text, r.therapist_text) for r in reloaded.transcript] == lines
    assert reloaded.generation_version == GENERATION_VERSION


@pytest.mark.parametrize("version", [None, 3, 5, GENERATION_VERSION + 1])
def test_a_session_from_another_generator_is_not_continued(sessions_dir, monkeypatch, version):
    record = _write_record(sessions_dir / "session_other", version)
    before = _snapshot(record)

    def no_provider(model, role):
        raise AssertionError("an agent was built for a session that must not continue")

    monkeypatch.setattr(dyad_mod, "build_agent", no_provider)

    with pytest.raises(session_mod.IncompatibleSessionError, match="generation_version"):
        _dyad(session_id="session_other")

    assert _snapshot(record) == before
    assert [p.name for p in sessions_dir.iterdir()] == ["session_other"]
    # Still readable, for scoring and analysis.
    assert resume_session("session_other").turn_count == 1


def test_a_session_without_its_opening_is_not_continued(sessions_dir, monkeypatch):
    record = _write_record(sessions_dir / "session_no_opening", GENERATION_VERSION, opening=None)
    before = _snapshot(record)
    monkeypatch.setattr(dyad_mod, "build_agent", lambda model, role: pytest.fail("agent built"))

    with pytest.raises(session_mod.IncompatibleSessionError, match="opening"):
        _dyad(session_id="session_no_opening")

    assert _snapshot(record) == before


def test_a_case_session_is_not_continued_with_a_supplied_prior(sessions_dir, monkeypatch):
    record = _write_record(sessions_dir / "session_case", GENERATION_VERSION)
    before = _snapshot(record)
    monkeypatch.setattr(dyad_mod, "build_agent", lambda model, role: pytest.fail("agent built"))

    with pytest.raises(session_mod.IncompatibleSessionError, match="supplied patient prior"):
        _dyad(session_id="session_case", patient_prior=PatientPrior(case_name=CASE))

    assert _snapshot(record) == before


def test_saving_an_old_record_keeps_its_generation_version(sessions_dir):
    record = _write_record(sessions_dir / "session_v3", 3)
    unversioned = _write_record(sessions_dir / "session_v1", None)

    resume_session("session_v3").save_metadata()
    resume_session("session_v1").save_metadata()

    assert json.loads((record / "metadata.json").read_text())["generation_version"] == 3
    assert "generation_version" not in json.loads((unversioned / "metadata.json").read_text())


def test_a_stub_session_writes_a_complete_v6_record(sessions_dir, agents):
    session = _dyad().run(n_turns=2)

    meta = json.loads((session.session_dir / "metadata.json").read_text())
    transcript = (session.session_dir / "transcript.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in transcript]

    assert GENERATION_VERSION == 6
    assert meta["generation_version"] == GENERATION_VERSION
    assert meta["turn_count"] == 2
    assert meta["initial_patient_prompt"] == OPENING
    assert [(r["turn"], r["patient_text"], r["therapist_text"]) for r in rows] == [
        (1, "patient 1", "therapist 2"),
        (2, "patient 3", "therapist 4"),
    ]
