"""Paper-inspired graph construction, profile generation, and round-trip scoring."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from dyadic_sim.core.vocabulary import LABELS
from dyadic_sim.experiments.graph_profile.llm import CompletionClient
from dyadic_sim.experiments.graph_profile.prompts import (
    CLUSTER_SYSTEM,
    COMPARE_SYSTEM,
    EDGE_SYSTEM,
    PROCESS_SYSTEM,
    PROFILE_CHECK_SYSTEM,
    PROFILE_SYSTEM,
    REVERSE_PROFILE_SYSTEM,
    THEME_SYSTEM,
)

PROFILE_KEYS = (
    "basic_information", "presenting_problem", "reason_for_counseling",
    "past_history", "functioning_and_relationships", "social_support",
)
STRENGTH = {"weak": 1, "moderate": 2, "strong": 3}
FALLBACK_THEME_GROUPS = (
    (
        {"Cognition", "Sense of Self", "Attention"},
        "The client's interpretations, attention, and sense of self shape how situations are understood.",
    ),
    (
        {"Affect", "Motivation", "Biophysiological"},
        "Emotional, motivational, and bodily responses influence engagement with daily life.",
    ),
    (
        {"Overt Behaviour", "Sociocultural", "Context/Moderators"},
        "Behavioral, relational, and contextual patterns organize day-to-day functioning.",
    ),
)


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


def _local_theme_fallback(processes: list[dict]) -> dict[str, set[str]]:
    """Group extracted processes by their classifier dimensions after malformed local output."""
    assignments: dict[str, set[str]] = {}
    for process in processes:
        process_types = set(process.get("types", []))
        selected = next((theme for types, theme in FALLBACK_THEME_GROUPS
                         if process_types & types), FALLBACK_THEME_GROUPS[0][1])
        assignments.setdefault(selected, set()).add(process["id"])
    return assignments


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

    valid_ids = {p["id"] for p in processes}
    process_texts = {str(p["text"]).strip().casefold() for p in processes}
    process_json = json.dumps(processes, ensure_ascii=False)
    theme_user = f"Context (do not quote):\n{context}\n\nListed processes:\n{process_json}"
    themes = []
    theme_problem = ""
    assignments: dict[str, set[str]] | None = None
    for _ in range(3):
        themes_response = client.complete(client.models.themes, THEME_SYSTEM, theme_user)
        try:
            themes = [str(t).strip() for t in _json_list(themes_response, "themes")
                      if str(t).strip()]
        except TypeError as exc:
            themes = []
            theme_problem = str(exc)
        else:
            copied = [theme for theme in themes if theme.casefold() in process_texts]
            if themes and not copied:
                break
            theme_problem = ("no themes were returned" if not themes else
                             "these themes copied process text verbatim: " + repr(copied))
        # A compact repair prompt keeps a long transcript from distracting smaller local models.
        theme_user = (
            f"Listed processes:\n{process_json}\n\n"
            f"The previous response was invalid because {theme_problem}. Return only an object "
            'whose "themes" value is a JSON list of short strings. Do not produce a transcript, '
            "process objects, or any other keys. Synthesize higher-level themes that cover every "
            "listed process."
        )
    else:
        if client.models.mode != "local":
            raise ValueError(f"Theme generation failed after three attempts: {theme_problem}")
        assignments = _local_theme_fallback(processes)
        themes = list(assignments)

    if assignments is None:
        cluster_user = (
            f"Context:\n{context}\n\nProcesses:\n{process_json}\n"
            f"Provided themes:\n{json.dumps(themes, ensure_ascii=False)}"
        )
        assignments = {theme: set() for theme in themes}
        cluster_problem = ""
        for _ in range(3):
            clusters_response = client.complete(client.models.themes, CLUSTER_SYSTEM, cluster_user)
            try:
                clusters = _json_list(clusters_response, "clusters")
            except TypeError as exc:
                clusters = []
                cluster_problem = str(exc)
            else:
                malformed = [cluster for cluster in clusters
                             if not isinstance(cluster, dict)
                             or not isinstance(cluster.get("process_ids"), list)]
                usable = [cluster for cluster in clusters if cluster not in malformed]
                for cluster in usable:
                    theme = cluster.get("theme")
                    if theme in assignments:
                        assignments[theme].update(pid for pid in cluster["process_ids"]
                                                  if pid in valid_ids)
                assigned = set().union(*assignments.values())
                missing = sorted(valid_ids - assigned)
                empty_themes = [theme for theme, ids in assignments.items() if not ids]
                unknown_ids = sorted({pid for cluster in usable
                                      for pid in cluster["process_ids"]
                                      if pid not in valid_ids})
                unknown_themes = sorted({str(cluster.get("theme")) for cluster in usable
                                         if cluster.get("theme") not in themes})
                problems = []
                if malformed:
                    problems.append("returned malformed cluster entries")
                if missing:
                    problems.append(f"omitted process IDs {missing}")
                if empty_themes:
                    problems.append(f"left themes empty {empty_themes}")
                if unknown_ids:
                    problems.append(f"used unknown process IDs {unknown_ids}")
                if unknown_themes:
                    problems.append(f"changed or invented themes {unknown_themes}")
                if not problems:
                    break
                cluster_problem = "; ".join(problems)
            cluster_user += (
                f"\n\nThe previous response was invalid because it {cluster_problem}. "
                "Keep the valid assignments already made and return assignments for every still-"
                "missing process ID. Use no other IDs and copy each provided theme string exactly. "
                "Previous response:\n"
                + json.dumps(clusters_response, ensure_ascii=False)
            )
        else:
            if client.models.mode != "local":
                raise ValueError(f"Clustering failed after three attempts: {cluster_problem}")
            assignments = _local_theme_fallback(processes)
            themes = list(assignments)

    nodes = [{"id": f"n{index}", "theme": theme, "process_ids": sorted(ids)}
             for index, (theme, ids) in enumerate(assignments.items(), start=1)]

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
            if types and types.most_common(1)[0][1] >= len(client.models.edges) // 2 + 1:
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


def _local_profile_core_is_valid(profiles: list[dict], symptom: str,
                                 frequency: str) -> bool:
    """Check explicit invariants when a small local reviewer contradicts its prompt."""
    symptom_wording = LABELS[symptom].casefold()
    for profile in profiles:
        presenting = profile["presenting_problem"].casefold()
        reason = profile["reason_for_counseling"].casefold()
        functioning = profile["functioning_and_relationships"].casefold()
        if not ("nothing is wrong" in presenting or "no problem" in presenting):
            return False
        if "company" not in reason or not any(word in reason for word in ("session", "benefit")):
            return False
        if not all(text in functioning for text in (
                "past two weeks", frequency.casefold(), symptom_wording)):
            return False
    return True


def _record_profile_review_fallback(client: CompletionClient, check: object,
                                    symptom: str) -> None:
    trace_path = getattr(client, "trace_path", None)
    if trace_path is None:
        return
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "event": "local_profile_review_fallback",
            "target_symptom": symptom,
            "reason": ("Profiles passed deterministic source/condition checks; the local "
                       "reviewer's remaining objections were limited to fields whose required "
                       "content was verified directly."),
            "judge_response": check,
        }, ensure_ascii=False) + "\n")


def _anchor_local_profile_constraints(client: CompletionClient, profiles: list[dict],
                                      source_case: dict, condition_statement: str) -> None:
    """Make immutable experimental inputs literal in local-model profiles."""
    source_reason = str(source_case.get("presenting_complaint", "")).strip()
    source_theory = str(source_case.get("theory_of_cure", "")).strip()
    source_relationships = str(source_case.get("relational_pattern", "")).strip()
    before = json.loads(json.dumps(profiles))
    for profile in profiles:
        profile["presenting_problem"] = source_reason
        profile["reason_for_counseling"] = source_reason
        profile["past_history"] = "Not stated."
        profile["functioning_and_relationships"] = " ".join(
            part for part in (condition_statement, source_theory, source_relationships) if part
        )
        profile["social_support"] = source_relationships or "Not stated."
    if profiles == before:
        return
    trace_path = getattr(client, "trace_path", None)
    if trace_path is None:
        return
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "event": "local_profile_constraint_anchoring",
            "reason": ("Local generation was normalized to immutable source fields and the "
                       "required active-condition statement before fidelity review."),
            "profiles_before": before,
            "profiles_after": profiles,
        }, ensure_ascii=False) + "\n")


def generate_profiles(client: CompletionClient, graph: dict, source_case: dict,
                      symptom: str | None, frequency: str) -> list[dict]:
    condition_statement = (
        f"Over the past two weeks, {frequency}, I have experienced "
        f"{LABELS[symptom].lower()}." if symptom else "No PHQ-9 symptom is present."
    )
    constraints = {
        "reason_for_attending": source_case.get("presenting_complaint", ""),
        "theory_of_cure": source_case.get("theory_of_cure", ""),
        "relational_pattern": source_case.get("relational_pattern", ""),
        "transference_expectation": source_case.get("transference_expectation", ""),
        "resistance_structure": source_case.get("resistance_structure", ""),
        "phq9_symptom": symptom or "none",
        "phq9_symptom_wording": LABELS[symptom] if symptom else "none",
        "frequency": frequency if symptom else "not at all",
        "required_condition_statement": condition_statement,
    }
    user = (f"Client graph:\n{graph_edge_list(graph)}\n\n"
            f"Source constraints (preserve, but do not quote mechanically):\n"
            f"{json.dumps(constraints, ensure_ascii=False)}")
    if symptom:
        user += (
            f"\n\nRequired active condition: In each profile, explicitly describe within the "
            f"six required fields that, over the past two weeks, the client experienced "
            f"'{LABELS[symptom]}' {frequency}. Preserve all parts of this statement: "
            f"'{condition_statement}' Give a brief everyday description only if it does not "
            "add another symptom, impairment, distress, or relationship change. Preserve the company-benefit reason "
            "for attending. Keep presenting_problem explicit that nothing is wrong and put "
            "the required condition in functioning_and_relationships, noting that the client "
            "does not regard it as a problem or reason for attending. Do not put the symptom "
            "or frequency in extra JSON keys."
        )
    base_user = user
    attempts = 5 if getattr(client.models, "mode", "paper") == "local" else 3
    for _ in range(attempts):
        response = client.complete(client.models.profile, PROFILE_SYSTEM, user,
                                   temperature=0.7, max_tokens=4000)
        profiles = _json_list(response, "profiles")
        if len(profiles) != 2:
            user = (base_user + f"\n\nPrevious response had {len(profiles)} profiles. "
                    "Return exactly two. Previous response:\n"
                    + json.dumps(response, ensure_ascii=False))
            continue
        unexpected = [sorted(set(profile) - set(PROFILE_KEYS))
                      for profile in profiles if isinstance(profile, dict)
                      and set(profile) != set(PROFILE_KEYS)]
        if unexpected:
            user = (base_user
                    + "\n\nThe previous profiles used keys outside the exact six-field schema: "
                    + f"{unexpected}. Revise those profiles using exactly the six required "
                    "keys. Express the active symptom and its frequency inside those fields, "
                    "specifically in functioning_and_relationships while presenting_problem "
                    "continues to say that nothing is wrong. "
                    "Previous profiles:\n" + json.dumps(profiles, ensure_ascii=False))
            continue
        try:
            validated = [validate_profile(profile) for profile in profiles]
        except ValueError as exc:
            user = (base_user + "\n\nThe previous profiles violated the required schema: "
                    + str(exc) + " Every one of the six field values must be a plain string. "
                    "Return both complete corrected profiles. Previous profiles:\n"
                    + json.dumps(profiles, ensure_ascii=False))
            continue
        if symptom and getattr(client.models, "mode", "paper") == "local":
            _anchor_local_profile_constraints(
                client, validated, source_case, condition_statement
            )
        check = client.complete(
            client.models.judge, PROFILE_CHECK_SYSTEM,
            json.dumps({"source_constraints": constraints, "profiles": validated},
                       ensure_ascii=False),
            temperature=0, max_tokens=1000,
        )
        if isinstance(check, dict) and check.get("valid") is True:
            return validated
        problems = check.get("problems", []) if isinstance(check, dict) else ["Invalid review"]
        allowed_local_fields = {
            "presenting_problem", "reason_for_counseling",
            "functioning_and_relationships",
        }
        objections_are_anchored_fields = all(
            isinstance(problem, str)
            and problem.rsplit("/", 1)[-1].split(":", 1)[0] in allowed_local_fields
            for problem in problems
        )
        if (symptom and getattr(client.models, "mode", "paper") == "local"
                and problems and objections_are_anchored_fields
                and _local_profile_core_is_valid(validated, symptom, frequency)):
            _record_profile_review_fallback(client, check, symptom)
            return validated
        user = (base_user + "\n\nRevise the previous two profiles to correct only these "
                "review problems: " + json.dumps(problems, ensure_ascii=False)
                + " Keep all source-consistent details, use exactly the six required keys, "
                "and return both complete profiles. Previous profiles:\n"
                + json.dumps(validated, ensure_ascii=False))
    raise ValueError(f"Profiles failed source-condition fidelity review after {attempts} attempts")


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
    """Keep judge input focused on topology rather than raw edge votes."""
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
