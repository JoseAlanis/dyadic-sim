"""Paper-inspired graph construction, profile generation, and round-trip scoring."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from graph_experiment.llm import CompletionClient
from graph_experiment.prompts import (
    CLUSTER_SYSTEM,
    COMPARE_SYSTEM,
    EDGE_SYSTEM,
    PROCESS_SYSTEM,
    PROFILE_CHECK_SYSTEM,
    PROFILE_SYSTEM,
    REVERSE_PROFILE_SYSTEM,
    THEME_SYSTEM,
)
from symptom_scoring.prior_vocabulary import LABELS

PROFILE_KEYS = (
    "basic_information", "presenting_problem", "reason_for_counseling",
    "past_history", "functioning_and_relationships", "social_support",
)
STRENGTH = {"weak": 1, "moderate": 2, "strong": 3}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def source_units(case: dict, symptom: str | None, frequency: str) -> list[dict]:
    units = []
    for key in ("presenting_complaint", "theory_of_cure", "relational_pattern",
                "transference_expectation", "resistance_structure"):
        for sentence in re.split(r"(?<=[.!?])\s+", str(case.get(key, "")).strip()):
            if sentence.strip():
                units.append({"source": key, "text": sentence.strip()})
    if symptom:
        units.append({"source": "phq9", "text": f"Over the past two weeks: {LABELS[symptom]} — {frequency}."})
    return units


def transcript_units(turns: list[dict]) -> list[dict]:
    return [{"source": f"turn_{row['turn']}", "text": row["patient_text"]}
            for row in turns if row.get("patient_text", "").strip()]


def _json_list(value: object, key: str) -> list:
    if not isinstance(value, dict) or not isinstance(value.get(key), list):
        raise TypeError(f"Expected JSON object with list key {key!r}: {value!r}")
    return value[key]


def construct_graph(client: CompletionClient, units: list[dict], context: str,
                    *, source_kind: str) -> dict:
    """MP1, MP2, MP3, then MP5 model-based ensemble, with saved intermediate data."""
    if not units:
        raise ValueError("No source units to graph")
    processes = []
    for index, unit in enumerate(units):
        neighborhood = units[max(0, index - 1):min(len(units), index + 2)]
        classified = client.complete(
            client.models.process, PROCESS_SYSTEM,
            f"Source kind: {source_kind}\nTarget passage: {unit['text']}\n"
            f"Context before/after: {json.dumps(neighborhood, ensure_ascii=False)}",
        )
        if not isinstance(classified, dict):
            raise TypeError("Process classifier returned a non-object")
        if classified.get("is_process"):
            for item in _json_list(classified, "processes"):
                if not isinstance(item, dict) or not str(item.get("text", "")).strip():
                    continue
                processes.append({"id": f"p{len(processes) + 1}",
                                  "text": str(item["text"]).strip(),
                                  "types": item.get("types", []),
                                  "source": unit["source"]})
    if not processes:
        if source_kind == "prior_plus_phq9" and any(u["source"] == "phq9" for u in units):
            raise ValueError("PHQ-9 symptom was omitted by process classification")
        # An empty network is a valid finding for the null bed.
        return {"source_kind": source_kind, "processes": [], "nodes": [], "edges": []}
    if (source_kind == "prior_plus_phq9" and any(u["source"] == "phq9" for u in units)
            and not any(p["source"] == "phq9" for p in processes)):
        raise ValueError("PHQ-9 symptom was omitted by process classification")

    themes_response = client.complete(
        client.models.themes, THEME_SYSTEM,
        f"Context (do not quote):\n{context}\n\nListed processes:\n"
        + json.dumps(processes, ensure_ascii=False),
    )
    themes = [str(t).strip() for t in _json_list(themes_response, "themes") if str(t).strip()]
    if not themes:
        raise ValueError("Theme generator returned no themes for nonempty processes")
    clusters_response = client.complete(
        client.models.themes, CLUSTER_SYSTEM,
        f"Context:\n{context}\n\nProcesses:\n{json.dumps(processes, ensure_ascii=False)}\n"
        f"Provided themes:\n{json.dumps(themes, ensure_ascii=False)}",
    )
    clusters = _json_list(clusters_response, "clusters")
    valid_ids = {p["id"] for p in processes}
    nodes = []
    for theme in themes:
        matching = [c for c in clusters if isinstance(c, dict) and c.get("theme") == theme]
        ids = [pid for c in matching for pid in c.get("process_ids", []) if pid in valid_ids]
        nodes.append({"id": f"n{len(nodes) + 1}", "theme": theme,
                      "process_ids": sorted(set(ids))})
    assigned = {pid for node in nodes for pid in node["process_ids"]}
    if assigned != valid_ids:
        raise ValueError(f"Clustering omitted processes: {sorted(valid_ids - assigned)}")

    edges = []
    for source in nodes:
        for target in nodes:
            if source["id"] == target["id"]:
                continue
            votes = []
            for model in client.models.edges:
                decision = client.complete(
                    model, EDGE_SYSTEM,
                    f"Theme A: {source['theme']}\n"
                    f"Theme B: {target['theme']}",
                )
                if not isinstance(decision, dict):
                    raise TypeError("Edge model returned a non-object")
                votes.append(decision)
            positive = [v for v in votes if str(v.get("connection")) == "1"
                        and v.get("type") in {"excitatory", "inhibitory"}]
            types = Counter(v["type"] for v in positive)
            if types and types.most_common(1)[0][1] >= 2:
                relation = types.most_common(1)[0][0]
                agreeing = [v for v in positive if v["type"] == relation]
                strongest = max(agreeing, key=lambda v: STRENGTH.get(v.get("strength"), 0))
                edges.append({"id": f"e{len(edges) + 1}", "source": source["id"],
                              "target": target["id"], "type": relation,
                              "strength": strongest.get("strength", "weak"),
                              "explanation": strongest.get("explanation", ""),
                              "votes": votes})
    return {"source_kind": source_kind, "processes": processes, "nodes": nodes,
            "edges": edges}


def graph_edge_list(graph: dict) -> str:
    labels = {node["id"]: node["theme"] for node in graph["nodes"]}
    if not graph["edges"]:
        return "Nodes: " + json.dumps(list(labels.values()), ensure_ascii=False) + "\nEdges: []"
    return "\n".join(
        f"({labels[e['source']]}, {e['type']} {e['strength']}, {labels[e['target']]})"
        for e in graph["edges"]
    )


def validate_profile(profile: object) -> dict[str, str]:
    if not isinstance(profile, dict) or any(not isinstance(profile.get(k), str) for k in PROFILE_KEYS):
        raise ValueError(f"Profile must have six string fields: {profile!r}")
    return {key: profile[key].strip() for key in PROFILE_KEYS}


def profile_text(profile: dict) -> str:
    return "\n".join(f"{key.replace('_', ' ').title()}: {profile[key]}" for key in PROFILE_KEYS)


def generate_profiles(client: CompletionClient, graph: dict, source_case: dict,
                      symptom: str | None, frequency: str) -> list[dict]:
    constraints = {
        "reason_for_attending": source_case.get("presenting_complaint", ""),
        "theory_of_cure": source_case.get("theory_of_cure", ""),
        "relational_pattern": source_case.get("relational_pattern", ""),
        "transference_expectation": source_case.get("transference_expectation", ""),
        "resistance_structure": source_case.get("resistance_structure", ""),
        "phq9_symptom": symptom or "none",
        "frequency": frequency if symptom else "not at all",
    }
    user = (f"Client graph:\n{graph_edge_list(graph)}\n\n"
            f"Source constraints (preserve, but do not quote mechanically):\n"
            f"{json.dumps(constraints, ensure_ascii=False)}")
    for _ in range(3):
        response = client.complete(client.models.profile, PROFILE_SYSTEM, user,
                                   temperature=0.7, max_tokens=4000)
        profiles = _json_list(response, "profiles")
        if len(profiles) != 2:
            user += f"\nPrevious response had {len(profiles)} profiles; return exactly two."
            continue
        validated = [validate_profile(profile) for profile in profiles]
        check = client.complete(
            client.models.judge, PROFILE_CHECK_SYSTEM,
            json.dumps({"source_constraints": constraints, "profiles": validated},
                       ensure_ascii=False),
            temperature=0, max_tokens=1000,
        )
        if isinstance(check, dict) and check.get("valid") is True:
            return validated
        user += "\nCorrect these problems in the next two profiles: " + json.dumps(
            check.get("problems", []) if isinstance(check, dict) else ["Invalid review"],
            ensure_ascii=False)
    raise ValueError("Profiles failed source-condition fidelity review after three attempts")


def reconstruct_profile(client: CompletionClient, graph: dict, turns: list[dict]) -> dict:
    response = client.complete(
        client.models.profile, REVERSE_PROFILE_SYSTEM,
        f"Inferred graph:\n{graph_edge_list(graph)}\n\nSession transcript:\n"
        + json.dumps(turns, ensure_ascii=False),
        temperature=0, max_tokens=2500,
    )
    return validate_profile(response)


def compare_round_trip(client: CompletionClient, source_graph: dict,
                       source_profile: dict, reverse_graph: dict,
                       reverse_profile: dict, symptom: str | None) -> dict:
    response = client.complete(
        client.models.judge, COMPARE_SYSTEM,
        json.dumps({"source_graph": comparison_view(source_graph),
                    "source_profile": source_profile,
                    "reconstructed_graph": comparison_view(reverse_graph),
                    "reconstructed_profile": reverse_profile,
                    "target_symptom": symptom or "no symptom"}, ensure_ascii=False),
        temperature=0, max_tokens=2500,
    )
    if not isinstance(response, dict):
        raise TypeError("Comparison judge returned a non-object")
    source_nodes = {node["id"] for node in source_graph["nodes"]}
    source_edges = {edge["id"] for edge in source_graph["edges"]}
    matched_nodes = set(response.get("matched_source_node_ids", [])) & source_nodes
    matched_edges = set(response.get("matched_source_edge_ids", [])) & source_edges
    fields = set(response.get("profile_supported_fields", [])) & set(PROFILE_KEYS)
    response["node_recall"] = len(matched_nodes) / len(source_nodes) if source_nodes else None
    response["edge_recall"] = len(matched_edges) / len(source_edges) if source_edges else None
    response["profile_field_coverage"] = len(fields) / len(PROFILE_KEYS)
    return response


def comparison_view(graph: dict) -> dict:
    """Keep judge input focused on topology rather than three raw model votes."""
    return {
        "nodes": [{"id": n["id"], "theme": n["theme"]} for n in graph["nodes"]],
        "edges": [{"id": e["id"], "source": e["source"], "target": e["target"],
                   "type": e["type"]} for e in graph["edges"]],
    }


def render_svg(graph: dict, path: Path) -> None:
    """Standalone graph image without adding a graph library dependency."""
    from matplotlib import pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    nodes = graph["nodes"]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.3, 1.3)
    ax.axis("off")
    positions = {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / max(1, len(nodes))
        positions[node["id"]] = (math.cos(angle), math.sin(angle))
    for edge in graph["edges"]:
        a, b = positions[edge["source"]], positions[edge["target"]]
        color = "#176b4d" if edge["type"] == "excitatory" else "#ac3d3d"
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.5, color=color, alpha=0.75,
                                     connectionstyle="arc3,rad=0.15", shrinkA=35, shrinkB=35))
    for node in nodes:
        x, y = positions[node["id"]]
        label = "\n".join(re.findall(r".{1,27}(?:\s|$)", node["theme"])[:4]) or node["theme"]
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "round,pad=0.6", "facecolor": "#e7eef4", "edgecolor": "#516879"})
    if not nodes:
        ax.text(0, 0, "No clinically relevant processes detected", ha="center")
    ax.set_title("Client psychological graph  •  green: excitatory  •  red: inhibitory")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
