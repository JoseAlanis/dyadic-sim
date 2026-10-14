"""High-value checks for graph voting, source isolation, and study design."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_graph_experiment as runner
import simulation.dyad as dyad_mod
import simulation.session as session_mod
from agents.base_agent import AgentResponse
from graph_experiment.llm import CompletionClient, Models
from graph_experiment.pipeline import PROFILE_KEYS, construct_graph, source_units
from priors.patient_prior import PatientPrior
from run_graph_experiment import selected_conditions
from symptom_scoring.prior_vocabulary import KEYS


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


def test_local_json_completion_requests_ollama_json_mode_and_traces_retry(tmp_path, monkeypatch):
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        content = "invalid JSON" if len(requests) == 1 else '{"is_process": false, "processes": []}'
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content),
                                     finish_reason="stop")],
            model=kwargs["model"], id=f"call-{len(requests)}",
            system_fingerprint=None, usage=None,
        )

    client = CompletionClient(Models.from_env(mode="local"), tmp_path / "trace.jsonl")
    monkeypatch.setattr(client, "_client", lambda _model: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))

    result = client.complete("qwen2.5:7b", "Return JSON only", "Classify this")

    assert result == {"is_process": False, "processes": []}
    assert len(requests) == 2
    assert all(request["response_format"] == {"type": "json_object"}
               for request in requests)
    trace = [json.loads(line) for line in client.trace_path.read_text().splitlines()]
    assert [entry["attempt"] for entry in trace] == [1, 2]
    assert all(entry["finish_reason"] == "stop" for entry in trace)


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
