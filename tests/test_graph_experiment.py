"""High-value checks for graph voting, source isolation, and study design."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import dyadic_sim.cli.run_graph_experiment as runner
import dyadic_sim.generation.dyad as dyad_mod
import dyadic_sim.generation.session as session_mod
from dyadic_sim.cli.run_graph_experiment import selected_conditions
from dyadic_sim.core.vocabulary import KEYS
from dyadic_sim.experiments.graph_profile.llm import (
    CompletionClient,
    Models,
    _parse_json_response,
)
from dyadic_sim.experiments.graph_profile.pipeline import (
    PROFILE_KEYS,
    construct_graph,
    generate_profiles,
    source_units,
)
from dyadic_sim.generation.agents.base_agent import AgentResponse
from dyadic_sim.generation.priors.patient_prior import PatientPrior


class FakeClient:
    def __init__(self):
        self.models = SimpleNamespace(process="process", themes="themes",
                                      edges=("a", "b", "c"))
        self.edge_calls = 0

    def complete(self, model, system, user, **kwargs):
        if "psychological process classifier" in system:
            return {"is_process": True, "processes": [{"text": user.split("Target passage: ")[1].split("\n")[0],
                                                        "types": ["Cognition"]}]}
        if "Generate clinically meaningful themes" in system:
            return {"themes": ["Theme A", "Theme B"]}
        if "classifying psychological processes under provided themes" in system:
            return {"clusters": [{"theme": "Theme A", "process_ids": ["p1"]},
                                 {"theme": "Theme B", "process_ids": ["p2"]}]}
        self.edge_calls += 1
        # Two models agree on A -> B; the third disagrees. Reverse has no majority.
        if self.edge_calls <= 3:
            return {"connection": 1, "type": "excitatory" if model != "c" else "inhibitory",
                    "strength": "moderate", "explanation": "Test link"}
        return {"connection": 0}


def test_graph_uses_directional_majority_vote():
    graph = construct_graph(FakeClient(),
                            [{"source": "prior", "text": "I plan carefully."},
                             {"source": "phq9", "text": "Sleep trouble nearly every day."}],
                            "context", source_kind="prior_plus_phq9")
    assert len(graph["nodes"]) == 2
    assert [(edge["source"], edge["target"], edge["type"])
            for edge in graph["edges"]] == [("n1", "n2", "excitatory")]
    assert len(graph["edges"][0]["votes"]) == 3


def test_single_local_edge_decision_can_create_edge():
    client = FakeClient()
    client.models.edges = ("qwen2.5:14b",)
    graph = construct_graph(client,
                            [{"source": "prior", "text": "I plan carefully."},
                             {"source": "phq9", "text": "Sleep trouble nearly every day."}],
                            "context", source_kind="prior_plus_phq9")
    assert graph["edges"]
    assert all(len(edge["votes"]) == 1 for edge in graph["edges"])


def test_graph_retries_copied_themes_and_incomplete_clustering():
    class RetryingClient:
        def __init__(self):
            self.models = SimpleNamespace(process="process", themes="themes", edges=())
            self.theme_calls = 0
            self.cluster_calls = 0

        def complete(self, model, system, user, **kwargs):
            if "psychological process classifier" in system:
                text = user.split("Target passage: ")[1].split("\n")[0]
                return {"is_process": True,
                        "processes": [{"text": text, "types": ["Cognition"]}]}
            if "Generate clinically meaningful themes" in system:
                self.theme_calls += 1
                if self.theme_calls == 1:
                    return {"themes": ["I plan carefully."]}
                return {"themes": ["The client uses deliberate routines to manage daily life."]}
            if "classifying psychological processes under provided themes" in system:
                self.cluster_calls += 1
                ids = ["p1"] if self.cluster_calls == 1 else ["p2"]
                return {"clusters": [{
                    "theme": "The client uses deliberate routines to manage daily life.",
                    "process_ids": ids,
                }]}
            raise AssertionError(f"Unexpected prompt: {system}")

    client = RetryingClient()
    graph = construct_graph(
        client,
        [{"source": "prior", "text": "I plan carefully."},
         {"source": "prior", "text": "I keep a regular schedule."}],
        "context",
        source_kind="prior_plus_phq9",
    )

    assert client.theme_calls == 2
    assert client.cluster_calls == 2
    assert graph["nodes"][0]["process_ids"] == ["p1", "p2"]


def test_local_graph_falls_back_when_theme_output_stays_malformed():
    class MalformedThemeClient:
        def __init__(self):
            self.models = SimpleNamespace(
                process="process", themes="themes", edges=(), mode="local"
            )
            self.theme_calls = 0

        def complete(self, model, system, user, **kwargs):
            if "psychological process classifier" in system:
                text = user.split("Target passage: ")[1].split("\n")[0]
                process_type = "Affect" if "frustrated" in text else "Cognition"
                return {
                    "is_process": True,
                    "processes": [{"text": text, "types": [process_type]}],
                }
            if "Generate clinically meaningful themes" in system:
                self.theme_calls += 1
                return {"themes": {"invented_transcript": "not a list"}}
            raise AssertionError("Clustering should not be needed for the local fallback")

    client = MalformedThemeClient()
    graph = construct_graph(
        client,
        [
            {"source": "turn_1", "text": "I think carefully."},
            {"source": "turn_2", "text": "I feel frustrated."},
        ],
        "context",
        source_kind="synthetic_transcript",
    )

    assert client.theme_calls == 3
    assert len(graph["nodes"]) == 2
    assert {process_id for node in graph["nodes"] for process_id in node["process_ids"]} == {
        "p1",
        "p2",
    }


def test_local_graph_falls_back_when_cluster_output_stays_malformed():
    class MalformedClusterClient:
        def __init__(self):
            self.models = SimpleNamespace(
                process="process", themes="themes", edges=(), mode="local"
            )
            self.cluster_calls = 0

        def complete(self, model, system, user, **kwargs):
            if "psychological process classifier" in system:
                text = user.split("Target passage: ")[1].split("\n")[0]
                return {
                    "is_process": True,
                    "processes": [{"text": text, "types": ["Cognition"]}],
                }
            if "Generate clinically meaningful themes" in system:
                return {"themes": ["Deliberate thinking guides everyday decisions."]}
            if "classifying psychological processes" in system:
                self.cluster_calls += 1
                return {"clusters": {"invented_turn": 11}}
            raise AssertionError(f"Unexpected prompt: {system}")

    client = MalformedClusterClient()
    graph = construct_graph(
        client,
        [
            {"source": "turn_1", "text": "I think carefully."},
            {"source": "turn_2", "text": "I make deliberate choices."},
        ],
        "context",
        source_kind="synthetic_transcript",
    )

    assert client.cluster_calls == 3
    assert len(graph["nodes"]) == 1
    assert graph["nodes"][0]["process_ids"] == ["p1", "p2"]


def test_active_symptom_must_be_inside_exact_profile_schema():
    class ProfileClient:
        def __init__(self):
            self.models = SimpleNamespace(profile="profile", judge="judge", mode="local")
            self.profile_calls = 0
            self.judge_calls = 0

        def complete(self, model, system, user, **kwargs):
            if model == "profile":
                self.profile_calls += 1
                profile = {key: "Source-consistent detail" for key in PROFILE_KEYS}
                if self.profile_calls == 1:
                    profile["phq9_symptom"] = "lack_of_pleasure"
                else:
                    assert "within the six required fields" in user
                    assert "put the required condition in functioning_and_relationships" in user
                    profile["presenting_problem"] = "Nothing is wrong."
                    profile["functioning_and_relationships"] = (
                        "For two weeks I have had little interest nearly every day."
                    )
                return {"profiles": [profile, profile]}
            self.judge_calls += 1
            return {"valid": True, "problems": []}

    client = ProfileClient()
    profiles = generate_profiles(
        client,
        {"nodes": [], "edges": []},
        {"presenting_complaint": "I am using a company-benefit session."},
        "lack_of_pleasure",
        "nearly every day",
    )

    assert client.profile_calls == 2
    assert client.judge_calls == 1
    assert set(profiles[0]) == set(PROFILE_KEYS)


def test_local_profile_review_fallback_is_narrow():
    class ContradictoryReviewerClient:
        def __init__(self):
            self.models = SimpleNamespace(profile="profile", judge="judge", mode="local")

        def complete(self, model, system, user, **kwargs):
            if model == "judge":
                return {"valid": False,
                        "problems": ["profiles/0/presenting_problem",
                                     "profiles/1/presenting_problem"]}
            profile = {key: "Source-consistent detail" for key in PROFILE_KEYS}
            profile["presenting_problem"] = "Nothing is wrong."
            profile["reason_for_counseling"] = "My company offered a benefit session."
            profile["functioning_and_relationships"] = (
                "Over the past two weeks, nearly every day, I have experienced little "
                "interest or pleasure in doing things."
            )
            return {"profiles": [profile, profile]}

    profiles = generate_profiles(
        ContradictoryReviewerClient(),
        {"nodes": [], "edges": []},
        {"presenting_complaint": "Nothing is wrong; this is a company session."},
        "lack_of_pleasure",
        "nearly every day",
    )

    assert len(profiles) == 2
    assert profiles[0]["presenting_problem"] == (
        "Nothing is wrong; this is a company session."
    )
    assert profiles[0]["reason_for_counseling"] == profiles[0]["presenting_problem"]


def test_profile_generation_retries_non_string_fields():
    class SchemaRetryClient:
        def __init__(self):
            self.models = SimpleNamespace(profile="profile", judge="judge", mode="paper")
            self.profile_calls = 0

        def complete(self, model, system, user, **kwargs):
            if model == "judge":
                return {"valid": True, "problems": []}
            self.profile_calls += 1
            profile = {key: "detail" for key in PROFILE_KEYS}
            if self.profile_calls == 1:
                profile["basic_information"] = {"age": 35}
            return {"profiles": [profile, profile]}

    client = SchemaRetryClient()
    profiles = generate_profiles(
        client, {"nodes": [], "edges": []}, {"presenting_complaint": "Nothing is wrong."},
        None, "nearly every day",
    )

    assert client.profile_calls == 2
    assert profiles[0]["basic_information"] == "detail"


def test_profile_prompt_does_not_include_original_case_or_symptom():
    prior = PatientPrior(case_name="no_complaint",
                         presenting_complaint="SOURCE PRIOR SENTINEL",
                         symptoms="SYMPTOM SENTINEL",
                         profile_text="PROFILE SENTINEL")
    prompt = prior.build_system_prompt()
    assert "PROFILE SENTINEL" in prompt
    assert "SOURCE PRIOR SENTINEL" not in prompt
    assert "SYMPTOM SENTINEL" not in prompt


def test_default_design_contains_control_and_all_nine_symptoms():
    assert selected_conditions("all") == ["no_symptoms", *KEYS]
    case = {"presenting_complaint": "Nothing is wrong."}
    assert all(unit["source"] != "phq9" for unit in source_units(case, None, "nearly every day"))
    assert source_units(case, KEYS[0], "nearly every day")[-1]["source"] == "phq9"


def test_local_mode_routes_every_model_to_ollama_even_with_cloud_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_MODEL_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for key in ("LOCAL_PROCESS_MODEL_ID", "LOCAL_EDGE_MODEL_ID",
                "LOCAL_PROFILE_MODEL_ID", "LOCAL_JUDGE_MODEL_ID"):
        monkeypatch.delenv(key, raising=False)
    models = Models.from_env(mode="local")
    assert models.process == models.themes == "qwen2.5:7b"
    assert models.edges == ("qwen2.5:14b",)
    assert models.profile == models.dialogue == "mistral-nemo"
    assert models.judge == "qwen2.5:7b"
    client = CompletionClient(models, tmp_path / "trace.jsonl")
    assert {client._provider(model) for model in models.required()} == {"ollama"}
    assert Models.from_env().mode == "paper"


def test_local_mode_rejects_ollama_cloud_model(monkeypatch):
    monkeypatch.setenv("LOCAL_PROFILE_MODEL_ID", "gpt-oss:120b-cloud")
    with pytest.raises(ValueError, match="not allowed in local mode"):
        Models.from_env(mode="local")


@pytest.mark.parametrize("suffix", ["'", r"\n"])
def test_local_json_parser_accepts_harmless_model_suffixes(suffix):
    assert _parse_json_response('{"is_process": false, "processes": []}' + suffix) == {
        "is_process": False, "processes": []}


def test_local_json_parser_rejects_multiple_values():
    with pytest.raises(ValueError):
        _parse_json_response('{"is_process": true}; {"is_process": false}')


def test_local_json_calls_request_ollama_json_mode(tmp_path, monkeypatch):
    request = {}

    class FakeCompletions:
        def create(self, **kwargs):
            request.update(kwargs)
            message = SimpleNamespace(content='{"answer": true}')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                                   model="qwen2.5:7b", id="response-id",
                                   system_fingerprint=None, usage=None)

    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    models = Models.from_env(mode="local")
    client = CompletionClient(models, tmp_path / "trace.jsonl")
    monkeypatch.setattr(client, "_client", lambda model: fake)

    assert client.complete(models.process, "Return JSON.", "Classify this.") == {
        "answer": True}
    assert request["response_format"] == {"type": "json_object"}


def test_batch_writes_two_linked_sessions_and_resumes(tmp_path, monkeypatch):
    class NoNetworkClient:
        def __init__(self, *args):
            pass

        def preflight(self):
            pass

    class StubAgent:
        model = "mistral-nemo"

        def complete(self, **kwargs):
            return AgentResponse(content="I am doing fine.", model=self.model)

    profile = {key: "I am doing fine." for key in PROFILE_KEYS}
    graph = {"source_kind": "prior_plus_phq9", "processes": [], "nodes": [], "edges": []}
    monkeypatch.setattr(runner, "CompletionClient", NoNetworkClient)
    monkeypatch.setattr(runner, "construct_graph", lambda *args, **kwargs: graph)
    monkeypatch.setattr(runner, "generate_profiles", lambda *args: [profile, profile])
    monkeypatch.setattr(runner, "reconstruct_profile", lambda *args: profile)
    monkeypatch.setattr(runner, "compare_round_trip", lambda *args: {"node_recall": None,
        "edge_recall": None, "profile_field_coverage": 1.0, "target_symptom_recovered": True})
    monkeypatch.setattr(runner, "render_svg", lambda graph, path: path.write_text("<svg/>"))
    monkeypatch.setattr(dyad_mod, "build_agent", lambda *args: StubAgent())
    monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path / "sessions")
    output = tmp_path / "experiment"
    monkeypatch.setenv("GRAPH_MODEL_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(sys, "argv", ["run_graph_experiment.py", "--output-dir", str(output),
                                   "--conditions", "no_symptoms", "--repeats", "1", "--turns", "2",
                                   "--model-mode", "local"])
    runner.main()
    runner.main()  # checkpointed rerun must not append more turns

    manifest = runner.read_json(output / "manifest.json")
    assert manifest["models"]["mode"] == "local"
    assert manifest["graph_model_endpoint"] == "http://localhost:11434/v1"

    comparisons = list(output.glob("no_symptoms/run_01/profile_*/comparison.json"))
    assert len(comparisons) == 2
    with (output / "summary.csv").open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 2
    for pointer in output.glob("no_symptoms/run_01/profile_*/session.json"):
        session_dir = runner.read_json(pointer)["session_dir"]
        assert len((Path(session_dir) / "transcript.jsonl").read_text().splitlines()) == 2


class PromptRecorder:
    """Stub dialogue agent that keeps every system prompt it is sent."""

    def __init__(self, model, role):
        self.model, self.role, self.system_prompts = model, role, []

    def complete(self, system_prompt, messages, **kwargs):
        self.system_prompts.append(system_prompt)
        return AgentResponse(content=f"{self.role} {len(messages)}", model=self.model)


def _graph_session(tmp_path, turns):
    """Run (or continue) one graph-profile dialogue through the runner's own path."""
    profile = {key: f"PROFILE {key}" for key in PROFILE_KEYS}
    return runner._run_dialogue(profile, tmp_path / "graph.json", "no_symptoms", 1, 1,
                                tmp_path / "run", Models(dialogue="stub"), "cbt", turns, 64)


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    built: list[PromptRecorder] = []

    def build(model, role):
        built.append(PromptRecorder(model, role))
        return built[-1]

    monkeypatch.setattr(dyad_mod, "build_agent", build)
    monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path / "sessions")
    return built


def test_the_graph_runner_continues_its_session_with_the_profile_prompt(tmp_path, recorded):
    _graph_session(tmp_path, turns=1)
    saved = _graph_session(tmp_path, turns=3)

    assert [row["turn"] for row in saved] == [1, 2, 3]
    resumed_patient = [agent for agent in recorded if agent.role == "patient"][-1]
    assert len(resumed_patient.system_prompts) == 2
    for prompt in resumed_patient.system_prompts:
        assert "## Your Intake Profile" in prompt and "PROFILE presenting_problem" in prompt
    session_dir = Path(runner.read_json(tmp_path / "run" / "session.json")["session_dir"])
    meta = runner.read_json(session_dir / "metadata.json")
    assert meta["patient_prompt_source"] == "graph_derived_profile_only"
    assert meta["turn_count"] == 3


def test_a_graph_session_is_not_continued_by_the_plain_cli(tmp_path, recorded, monkeypatch):
    import dyadic_sim.cli.run as run_cli

    _graph_session(tmp_path, turns=1)
    session_dir = Path(runner.read_json(tmp_path / "run" / "session.json")["session_dir"])
    before = {p.name: p.read_bytes() for p in sorted(session_dir.iterdir())}
    monkeypatch.setattr(dyad_mod, "build_agent",
                        lambda *args: pytest.fail("agent built for a session that must not continue"))
    monkeypatch.setattr(sys, "argv", ["run.py", "--resume", str(session_dir),
                                      "--additional-turns", "2"])

    with pytest.raises(SystemExit) as stop:
        run_cli.main()

    assert stop.value.code == 1
    assert {p.name: p.read_bytes() for p in sorted(session_dir.iterdir())} == before
