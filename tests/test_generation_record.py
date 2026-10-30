"""What a session records about each completion, and that recording sends nothing new.

Stub providers stand in for Ollama, OpenAI and Anthropic. The requests the
wrappers send are pinned; what they report back (usage, finish reason, model
facts the provider confirmed) must survive saving and resuming, and anything
not reported stays unknown (None), never 0. The message lists themselves are
pinned in test_conversation_history.py.

    uv run pytest tests/test_generation_record.py -q
"""

from __future__ import annotations

import http.client
import io
import json
import urllib.error
from types import SimpleNamespace

import pytest

import dyadic_sim.generation.agents.local_agent as local_mod
import dyadic_sim.generation.dyad as dyad_mod
import dyadic_sim.generation.session as session_mod
from dyadic_sim.generation.agents.base_agent import AgentResponse, BaseAgent, Message
from dyadic_sim.generation.agents.claude_agent import ClaudeAgent
from dyadic_sim.generation.agents.local_agent import LocalAgent
from dyadic_sim.generation.agents.openai_agent import OpenAIAgent
from dyadic_sim.generation.dyad import Dyad
from dyadic_sim.generation.opening import FIRST_SESSION_OPENING
from dyadic_sim.generation.session import (
    PROMPTS_FILE,
    IncompatibleSessionError,
    Session,
    resume_session,
    sha256_text,
)
from dyadic_sim.generation.utterance import GENERATION_VERSION, PROMPT_VARIANT

CASE = "no_complaint"
SYSTEM = "You are a stub."
MESSAGES = [Message("user", "Hello."), Message("assistant", "Hi."), Message("user", "And?")]
CHAT = [{"role": m.role, "content": m.content} for m in MESSAGES]
UNKNOWN = {"response_model": None, "digest": None, "quantization": None, "context_length": None}
PS = {"models": [{
    "name": "mistral-nemo:latest",
    "model": "mistral-nemo:latest",
    "digest": "e7e06d10",
    "details": {"quantization_level": "Q4_0"},
    "context_length": 4096,
}]}


# --- Provider wrappers -------------------------------------------------------


def _chat_response(usage=True, finish="stop"):
    """An OpenAI-style chat completion, as Ollama's compatible endpoint returns it."""
    return SimpleNamespace(
        model="served-name",
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=7) if usage else None,
    )


def _chat_client(response):
    """A client whose chat.completions.create() keeps every call."""
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return response

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


def _serve(monkeypatch, payload):
    """Answer the wrapper's /api/ps query with `payload`, or raise it."""
    seen: list[str] = []

    def urlopen(url, timeout):
        seen.append(url)
        if isinstance(payload, Exception):
            raise payload
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(local_mod.urllib.request, "urlopen", urlopen)
    return seen


def test_the_ollama_wrapper_sends_the_same_request_and_reports_what_came_back(monkeypatch):
    agent = LocalAgent("mistral-nemo", "patient", base_url="http://ollama.test")
    agent._client, calls = _chat_client(_chat_response(finish="length"))
    seen = _serve(monkeypatch, PS)

    response = agent.complete(SYSTEM, MESSAGES, max_tokens=300)

    assert calls == [{
        "model": "mistral-nemo",
        "messages": [{"role": "system", "content": SYSTEM}, *CHAT],
        "max_tokens": 300,
        "temperature": 0.7,
    }]
    assert response.request == {"model": "mistral-nemo", "max_tokens": 300, "temperature": 0.7}
    assert (response.input_tokens, response.output_tokens) == (120, 7)
    assert response.finish_reason == "length"
    assert response.observed == {
        "response_model": "served-name",
        "digest": "e7e06d10",
        "quantization": "Q4_0",
        "context_length": 4096,
    }
    assert seen == ["http://ollama.test/api/ps"]


MALFORMED_ENTRY = {
    "name": "mistral-nemo:latest",
    "digest": 5,
    "details": "Q4_0",
    "context_length": "4096",
}


@pytest.mark.parametrize(
    "server",
    [
        urllib.error.URLError("unreachable"),
        http.client.IncompleteRead(b"{"),
        {"models": []},
        {"unexpected": "shape"},
        {"models": None},
        ["not", "an", "object"],
        {"models": [MALFORMED_ENTRY]},
    ],
    ids=["unreachable", "cut-off", "model-not-loaded", "unexpected-reply",
         "models-null", "reply-not-an-object", "malformed-entry"],
)
def test_what_ollama_does_not_report_stays_unknown(monkeypatch, server):
    agent = LocalAgent("mistral-nemo", "patient", base_url="http://ollama.test")
    agent._client, _ = _chat_client(_chat_response(usage=False, finish=None))
    _serve(monkeypatch, server)

    response = agent.complete(SYSTEM, MESSAGES)

    assert (response.input_tokens, response.output_tokens, response.finish_reason) == (None, None, None)
    assert response.observed == {**UNKNOWN, "response_model": "served-name"}


def test_the_openai_wrapper_sends_the_same_request(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = OpenAIAgent("gpt-stub", "therapist")
    agent._client, calls = _chat_client(_chat_response())

    response = agent.complete(SYSTEM, MESSAGES, max_tokens=300)

    assert calls == [{
        "model": "gpt-stub",
        "messages": [{"role": "system", "content": SYSTEM}, *CHAT],
        "max_tokens": 300,
        "temperature": 0.7,
    }]
    assert (response.input_tokens, response.output_tokens, response.finish_reason) == (120, 7, "stop")
    assert response.observed == {**UNKNOWN, "response_model": "served-name"}


def test_the_anthropic_wrapper_sends_the_same_request(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    agent = ClaudeAgent("claude-stub", "therapist")
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            model="claude-stub-served",
            content=[SimpleNamespace(text="ok")],
            usage=SimpleNamespace(input_tokens=50, output_tokens=5),
            stop_reason="end_turn",
        )

    agent._client = SimpleNamespace(messages=SimpleNamespace(create=create))

    response = agent.complete(SYSTEM, MESSAGES, max_tokens=300)

    assert calls == [{
        "model": "claude-stub",
        "system": SYSTEM,
        "messages": CHAT,
        "max_tokens": 300,
        "temperature": 0.7,
    }]
    assert (response.input_tokens, response.output_tokens, response.finish_reason) == (50, 5, "end_turn")
    assert response.observed == {**UNKNOWN, "response_model": "claude-stub-served"}


# --- The session record ------------------------------------------------------


class ReportingAgent(BaseAgent):
    """Answers "<role> <number of messages>" and reports diagnostics like a wrapper."""

    def __init__(self, model: str, role: str, usage: bool = True):
        super().__init__(model, role)
        self.usage = usage
        self.requests: list[tuple[str, list[list[str]]]] = []

    def complete(self, system_prompt, messages, max_tokens=512, temperature=0.7):
        self.requests.append((system_prompt, [[m.role, m.content] for m in messages]))
        return AgentResponse(
            content=f"{self.role} {len(messages)}",
            model=self.model,
            input_tokens=10 * len(messages) if self.usage else None,
            output_tokens=2 if self.usage else None,
            provider="stub",
            finish_reason="stop",
            request={"model": self.model, "max_tokens": max_tokens, "temperature": temperature},
            observed=dict(UNKNOWN),
        )

    def is_available(self) -> bool:
        return True


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)
    return tmp_path


def _install(monkeypatch, usage=True) -> list[ReportingAgent]:
    built: list[ReportingAgent] = []

    def build(model, role):
        built.append(ReportingAgent(model, role, usage=usage))
        return built[-1]

    monkeypatch.setattr(dyad_mod, "build_agent", build)
    return built


@pytest.fixture
def agents(monkeypatch):
    return _install(monkeypatch)


def _dyad(**kwargs) -> Dyad:
    return Dyad("stub", "stub", CASE, orientation="cbt", **kwargs)


def _rows(directory) -> list[dict]:
    lines = (directory / "transcript.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def _meta(directory) -> dict:
    return json.loads((directory / "metadata.json").read_text())


def _prompts(directory) -> list[dict]:
    return json.loads((directory / PROMPTS_FILE).read_text())["prompts"]


def _latest(built, role) -> ReportingAgent:
    return [agent for agent in built if agent.role == role][-1]


def test_each_completion_is_recorded_with_what_it_was_sent(sessions_dir, agents):
    session = _dyad().run(n_turns=2)
    rows = _rows(session.session_dir)
    patient, therapist = _latest(agents, "patient"), _latest(agents, "therapist")

    for index, row in enumerate(rows):
        turn = index + 1
        for role, agent in (("patient", patient), ("therapist", therapist)):
            record = row[f"{role}_diagnostics"]
            system, sent = agent.requests[index]
            assert (record["role"], record["turn"]) == (role, turn)
            assert record["requested"] == {"model": "stub", "max_tokens": 300, "temperature": 0.7}
            assert record["system_prompt_sha256"] == sha256_text(system)
            assert record["messages_sha256"] == sha256_text(json.dumps(sent, ensure_ascii=False))
            assert record["n_messages"] == len(sent)
            assert record["finish_reason"] == "stop"
            assert record["input_tokens"] == 10 * len(sent)
            assert record["output_tokens"] == 2
            assert record["observed"] == UNKNOWN

    meta = _meta(session.session_dir)
    assert (meta["generation_version"], meta["prompt_variant"]) == (GENERATION_VERSION, PROMPT_VARIANT)


def test_the_prompts_are_stored_as_text_with_their_hashes(sessions_dir, agents):
    session = _dyad().run(n_turns=2)
    entries = _prompts(session.session_dir)
    by_role: dict[str, list[dict]] = {}
    for entry in entries:
        by_role.setdefault(entry["role"], []).append(entry)

    # One patient prompt; one therapist prompt per agenda step it was sent.
    assert set(by_role) == {"opening", "patient_system", "therapist_system"}
    assert [e["text"] for e in by_role["opening"]] == [FIRST_SESSION_OPENING]
    assert [e["text"] for e in by_role["patient_system"]] == [
        _latest(agents, "patient").requests[0][0]]
    assert [e["text"] for e in by_role["therapist_system"]] == [
        system for system, _ in _latest(agents, "therapist").requests]
    assert [e["first_turn"] for e in by_role["therapist_system"]] == [1, 2]
    for entry in entries:
        assert entry["sha256"] == sha256_text(entry["text"])
    assert _meta(session.session_dir)["record_diagnostics"] == {
        "schema_version": 1,
        "since_turn": 1,
        "prompts_file": PROMPTS_FILE,
    }


def test_diagnostics_survive_saving_and_resuming(sessions_dir, agents):
    session = _dyad().run(n_turns=2)

    resumed = resume_session(session.session_id)

    for role in ("patient", "therapist"):
        saved = [row[f"{role}_diagnostics"] for row in _rows(session.session_dir)]
        assert [getattr(r, f"{role}_diagnostics") for r in resumed.transcript] == saved
        assert [getattr(r, f"{role}_diagnostics") for r in session.transcript] == saved


def test_unreported_usage_is_saved_as_unknown_not_zero(sessions_dir, monkeypatch):
    _install(monkeypatch, usage=False)

    session = _dyad().run(n_turns=1)
    row = _rows(session.session_dir)[0]

    for role in ("patient", "therapist"):
        assert row[f"{role}_diagnostics"]["input_tokens"] is None
        assert row[f"{role}_diagnostics"]["output_tokens"] is None
        assert row[f"{role}_tokens"] is None


def _write_record_without_diagnostics(directory):
    """A one-turn session of this generator, saved without diagnostics."""
    directory.mkdir()
    meta = {
        "metadata_schema_version": 2,
        "generation_version": GENERATION_VERSION,
        "prompt_variant": PROMPT_VARIANT,
        "session_id": directory.name,
        "patient_id": None,
        "therapist_model": "stub",
        "patient_model": "stub",
        "case_name": CASE,
        "orientation": "cbt",
        "patient_symptoms": "",
        "patient_symptom_levels": {},
        "conversation_language": "en",
        "initial_patient_prompt": FIRST_SESSION_OPENING,
        "turn_count": 1,
        "created_at": "2026-10-28T22:00:00",
    }
    (directory / "metadata.json").write_text(json.dumps(meta))
    row = {
        "turn": 1,
        "therapist_text": "What brings you here?",
        "patient_text": "The garage, mostly.",
        "therapist_model": "stub",
        "patient_model": "stub",
        "therapist_tokens": 12,
        "patient_tokens": 9,
    }
    (directory / "transcript.jsonl").write_text(json.dumps(row) + "\n")
    return directory


def test_turns_from_before_diagnostics_stay_unknown_after_resuming(sessions_dir, agents):
    legacy = _write_record_without_diagnostics(sessions_dir / "session_legacy")
    first_row = (legacy / "transcript.jsonl").read_text()

    _dyad(session_id="session_legacy").run(n_turns=1)

    rows = _rows(legacy)
    assert (legacy / "transcript.jsonl").read_text().startswith(first_row)
    assert "patient_diagnostics" not in rows[0]
    assert rows[1]["patient_diagnostics"]["turn"] == 2
    assert _meta(legacy)["record_diagnostics"]["since_turn"] == 2
    assert {entry["first_turn"] for entry in _prompts(legacy)} == {2}
    replayed = resume_session("session_legacy").transcript
    assert replayed[0].patient_diagnostics is None
    assert replayed[0].therapist_diagnostics is None


def test_resuming_never_rewrites_the_stored_prompts(sessions_dir, agents):
    session = _dyad().run(n_turns=1)
    before = _prompts(session.session_dir)

    _dyad(session_id=session.session_id).run(n_turns=1)

    after = _prompts(session.session_dir)
    # Earlier entries stay as they were; only turn 2's new agenda step is added.
    assert after[:len(before)] == before
    assert [(e["role"], e["first_turn"]) for e in after[len(before):]] == [("therapist_system", 2)]
    assert _meta(session.session_dir)["record_diagnostics"]["since_turn"] == 1


def test_a_changed_prompt_is_added_and_the_earlier_one_kept(sessions_dir):
    Session("session_x", "stub", "stub", CASE, "cbt").record_prompt("patient_system", "A", 1)
    Session("session_x", "stub", "stub", CASE, "cbt").record_prompt("patient_system", "B", 3)
    Session("session_x", "stub", "stub", CASE, "cbt").record_prompt("patient_system", "A", 5)

    stored = _prompts(sessions_dir / "session_x")
    assert [(e["text"], e["first_turn"]) for e in stored] == [("A", 1), ("B", 3)]


def _edit_json(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def _drop_therapist_prompts(store):
    store["prompts"] = [e for e in store["prompts"] if e["role"] != "therapist_system"]


def _alter_a_text(store):
    store["prompts"][1]["text"] += " (edited)"


def _strip_therapist_diagnostics(directory):
    rows = _rows(directory)
    del rows[0]["therapist_diagnostics"]
    (directory / "transcript.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


TAMPERING = {
    "store-missing": lambda d: (d / PROMPTS_FILE).unlink(),
    "store-unreadable": lambda d: (d / PROMPTS_FILE).write_text("{not json"),
    "store-other-schema": lambda d: _edit_json(d / PROMPTS_FILE, lambda s: s.update(schema_version=99)),
    "text-not-matching-hash": lambda d: _edit_json(d / PROMPTS_FILE, _alter_a_text),
    "turn-prompt-not-stored": lambda d: _edit_json(d / PROMPTS_FILE, _drop_therapist_prompts),
    "turn-without-diagnostics": _strip_therapist_diagnostics,
    "mark-malformed": lambda d: _edit_json(
        d / "metadata.json", lambda m: m["record_diagnostics"].update(since_turn="1")),
    "mark-removed": lambda d: _edit_json(d / "metadata.json", lambda m: m.pop("record_diagnostics")),
    "mark-null": lambda d: _edit_json(d / "metadata.json", lambda m: m.update(record_diagnostics=None)),
    "mark-in-the-future": lambda d: _edit_json(
        d / "metadata.json", lambda m: m["record_diagnostics"].update(since_turn=999)),
    "mark-at-zero": lambda d: _edit_json(
        d / "metadata.json", lambda m: m["record_diagnostics"].update(since_turn=0)),
    "mark-after-recorded-diagnostics": lambda d: _edit_json(
        d / "metadata.json", lambda m: m["record_diagnostics"].update(since_turn=2)),
}


@pytest.mark.parametrize("tamper", TAMPERING.values(), ids=TAMPERING.keys())
def test_a_session_whose_prompts_are_unaccounted_for_is_not_continued(
        sessions_dir, agents, monkeypatch, tamper):
    session = _dyad().run(n_turns=1)
    tamper(session.session_dir)
    before = {p.name: p.read_bytes() for p in session.session_dir.iterdir()}
    monkeypatch.setattr(dyad_mod, "build_agent", lambda model, role: pytest.fail("agent built"))

    with pytest.raises(IncompatibleSessionError, match="cannot be continued"):
        _dyad(session_id=session.session_id)

    assert {p.name: p.read_bytes() for p in session.session_dir.iterdir()} == before


def test_a_failed_prompt_write_is_retried_not_assumed_written(sessions_dir, agents, monkeypatch):
    dyad = _dyad()
    real_replace = session_mod.os.replace
    writes = []

    def replace_failing_third(source, target):
        writes.append(target)
        if len(writes) == 3:  # opening, patient prompt, then the therapist prompt
            raise OSError("disk full")
        return real_replace(source, target)

    monkeypatch.setattr(session_mod.os, "replace", replace_failing_third)

    with pytest.raises(OSError, match="disk full"):
        dyad.run(n_turns=1)
    session = dyad.run(n_turns=1)

    sent = _rows(session.session_dir)[0]["therapist_diagnostics"]["system_prompt_sha256"]
    assert ("therapist_system", sent) in {(e["role"], e["sha256"]) for e in _prompts(session.session_dir)}
    assert not list(session.session_dir.glob("*.partial"))


def test_an_unreadable_prompt_store_is_refused_not_overwritten(sessions_dir):
    session = Session("session_x", "stub", "stub", CASE, "cbt")
    path = session.session_dir / PROMPTS_FILE
    path.write_text("{not json")

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        session.record_prompt("opening", "Hello", 1)
    assert path.read_text() == "{not json"


# --- Sampling settings ------------------------------------------------------


def _requested(directory, role) -> list[dict]:
    return [row[f"{role}_diagnostics"]["requested"] for row in _rows(directory)]


@pytest.mark.parametrize(
    ("kwargs", "patient", "therapist"),
    [
        ({}, 0.7, 0.7),
        ({"temperature": 0.3}, 0.3, 0.3),
        ({"temperature": 0.3, "patient_temperature": 0.5}, 0.5, 0.3),
        ({"therapist_temperature": 0.2}, 0.7, 0.2),
    ],
    ids=["default", "both", "patient-override", "therapist-override"],
)
def test_each_role_is_sent_the_temperature_chosen_for_it(sessions_dir, agents, kwargs, patient, therapist):
    session = _dyad().run(n_turns=2, **kwargs)

    assert _requested(session.session_dir, "patient") == [
        {"model": "stub", "max_tokens": 300, "temperature": patient}] * 2
    assert _requested(session.session_dir, "therapist") == [
        {"model": "stub", "max_tokens": 300, "temperature": therapist}] * 2


def test_the_provider_request_is_unchanged_by_default_and_carries_a_chosen_temperature(
        sessions_dir, monkeypatch):
    sent: list[dict] = []

    def create(**kwargs):
        sent.append(kwargs)
        return _chat_response()

    def build(model, role):
        agent = LocalAgent(model, role, base_url="http://ollama.test")
        agent._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return agent

    monkeypatch.setattr(dyad_mod, "build_agent", build)
    _serve(monkeypatch, PS)

    _dyad().run(n_turns=1)
    _dyad().run(n_turns=1, temperature=0.3)

    assert [set(call) for call in sent] == [{"model", "messages", "max_tokens", "temperature"}] * 4
    assert [(c["max_tokens"], c["temperature"]) for c in sent] == [(300, 0.7)] * 2 + [(300, 0.3)] * 2


def _files(directory) -> dict:
    return {p.name: p.read_bytes() for p in directory.iterdir()}


@pytest.mark.parametrize(
    ("therapist_model", "run_kwargs"),
    [
        ("stub", {"temperature": 0.3}),
        ("stub", {"max_tokens": 200}),
        ("stub", {"patient_temperature": 0.5}),
        ("other-model", {}),
    ],
    ids=["temperature", "max-tokens", "one-role-temperature", "model"],
)
def test_a_session_is_not_continued_with_other_settings(sessions_dir, agents, therapist_model, run_kwargs):
    session = _dyad().run(n_turns=1)
    before = _files(session.session_dir)
    resumed = Dyad(therapist_model, "stub", CASE, orientation="cbt", session_id=session.session_id)

    with pytest.raises(IncompatibleSessionError, match="same settings"):
        resumed.run(n_turns=1, **run_kwargs)

    assert _files(session.session_dir) == before
    assert not resumed.patient_agent.requests and not resumed.therapist_agent.requests


def test_a_session_is_not_continued_when_its_prompt_renders_differently(
        sessions_dir, agents, monkeypatch):
    session = _dyad().run(n_turns=1)
    before = _files(session.session_dir)
    resumed = _dyad(session_id=session.session_id)
    # As if the case file had been edited after the first run.
    monkeypatch.setattr(type(resumed.patient_prior), "build_system_prompt",
                        lambda self: "An edited case.")

    with pytest.raises(IncompatibleSessionError, match="renders differently"):
        resumed.run(n_turns=1)

    assert _files(session.session_dir) == before
    assert not resumed.patient_agent.requests


def test_the_same_settings_continue(sessions_dir, agents):
    session = _dyad().run(n_turns=1, temperature=0.3)

    _dyad(session_id=session.session_id).run(n_turns=1, temperature=0.3)

    assert _requested(session.session_dir, "patient")[1]["temperature"] == 0.3


def test_the_batch_runner_passes_the_temperatures_to_each_session(monkeypatch):
    from argparse import Namespace

    import dyadic_sim.cli.run_symptom_experiments as batch
    from dyadic_sim.cli.sampling import temperature_flags

    commands = []
    monkeypatch.setattr(batch.subprocess, "run", lambda cmd, check: commands.append(cmd))
    chosen = Namespace(temperature=0.3, patient_temperature=None, therapist_temperature=0.5)

    batch.run_session("stub", "stub", "case_x", "cbt", 10, 300, sampling=temperature_flags(chosen))

    assert commands[0][-4:] == ["--temperature", "0.3", "--therapist-temperature", "0.5"]


def test_a_refused_resume_with_a_new_patient_id_writes_nothing(sessions_dir, agents):
    session = _dyad().run(n_turns=1, temperature=0.3)
    before = _files(session.session_dir)
    resumed = _dyad(session_id=session.session_id, patient_id="P-new")

    with pytest.raises(IncompatibleSessionError, match="same settings"):
        resumed.run(n_turns=1)

    assert _files(session.session_dir) == before


def test_an_accepted_resume_saves_the_new_patient_id(sessions_dir, agents):
    session = _dyad().run(n_turns=1)

    _dyad(session_id=session.session_id, patient_id="P-new").run(n_turns=1)

    assert _meta(session.session_dir)["patient_id"] == "P-new"
