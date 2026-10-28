"""Graph-first PHQ-9 round-trip study.

Default: 10 conditions x 5 independent graphs x 2 profiles/sessions = 100 sessions.
All output is checkpointed under data/graph_experiments/<run-id>/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from dyadic_sim.core import paths
from dyadic_sim.core.vocabulary import KEYS
from dyadic_sim.experiments.graph_profile.llm import CompletionClient, Models
from dyadic_sim.experiments.graph_profile.pipeline import (
    compare_round_trip,
    construct_graph,
    generate_profiles,
    profile_text,
    read_json,
    reconstruct_profile,
    render_svg,
    source_units,
    transcript_units,
    write_json,
)
from dyadic_sim.generation.dyad import Dyad
from dyadic_sim.generation.priors.loader import load_patient_case
from dyadic_sim.generation.priors.patient_prior import PatientPrior

paths.load_env()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/graph_experiments/no_complaint_phq9_graph_v1"))
    parser.add_argument("--conditions", default="all",
                        help="Comma-separated PHQ-9 keys or 'all' (includes no_symptoms)")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--frequency", choices=["several days", "more than half the days",
                                                "nearly every day"], default="nearly every day")
    parser.add_argument("--orientation", default="cbt")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--model-mode", choices=["paper", "local"], default="paper",
                        help="Paper model routing or a no-API local Ollama adaptation")
    parser.add_argument("--check", action="store_true", help="Check model availability only")
    return parser.parse_args()


def selected_conditions(raw: str) -> list[str]:
    if raw == "all":
        return ["no_symptoms", *KEYS]
    chosen = [x.strip() for x in raw.split(",") if x.strip()]
    invalid = set(chosen) - {"no_symptoms", *KEYS}
    if invalid or not chosen or len(chosen) != len(set(chosen)):
        raise ValueError(f"Invalid or duplicate conditions: {raw}")
    return chosen


def _load_turns(session_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (session_dir / "transcript.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]


def _run_dialogue(profile: dict, graph_path: Path, condition: str, repeat: int,
                  profile_number: int, folder: Path, models: Models,
                  orientation: str, turns: int, max_tokens: int) -> list[dict]:
    pointer = folder / "session.json"
    prior = PatientPrior(case_name="no_complaint", profile_text=profile_text(profile))
    if pointer.exists():
        session_id = read_json(pointer)["session_id"]
        dyad = Dyad(models.dialogue, models.dialogue, "no_complaint",
                    orientation=orientation, session_id=session_id, patient_prior=prior)
    else:
        dyad = Dyad(models.dialogue, models.dialogue, "no_complaint",
                    orientation=orientation, patient_prior=prior,
                    patient_id=f"{condition}_r{repeat:02d}_p{profile_number:02d}")
        write_json(pointer, {"session_id": dyad.session.session_id,
                             "session_dir": str(dyad.session.session_dir.resolve())})
        dyad.session.save_metadata({
            "experiment": "no_complaint_phq9_graph_v1", "condition": condition,
            "repeat": repeat, "profile_number": profile_number,
            "source_graph_path": str(graph_path.resolve()),
            "profile_path": str((folder / "profile.json").resolve()),
            "patient_prompt_source": "graph_derived_profile_only",
        })
    remaining = turns - dyad.session.turn_count
    if remaining < 0:
        raise ValueError(f"Session exceeds requested {turns} turns: {dyad.session.session_id}")
    if remaining:
        dyad.run(n_turns=remaining, max_tokens=max_tokens)
    session_dir = Path(read_json(pointer)["session_dir"])
    saved_turns = _load_turns(session_dir)
    if len(saved_turns) != turns:
        raise ValueError(f"Expected {turns} saved turns, found {len(saved_turns)}")
    return saved_turns


def _summary(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.glob("*/run_*/profile_*/comparison.json")):
        comparison = read_json(path)
        profile_dir = path.parent
        rows.append({
            "condition": path.parts[-4], "repeat": path.parts[-3],
            "profile": path.parts[-2],
            "node_recall": comparison.get("node_recall"),
            "edge_recall": comparison.get("edge_recall"),
            "profile_field_coverage": comparison.get("profile_field_coverage"),
            "target_symptom_recovered": comparison.get("target_symptom_recovered"),
            "profile_contradictions": len(comparison.get("profile_contradictions", [])),
            "other_phq9_symptoms_invented": len(comparison.get("other_phq9_symptoms_invented", [])),
            "session_dir": read_json(profile_dir / "session.json")["session_dir"],
        })
    if rows:
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = arguments()
    if args.repeats < 1 or args.turns < 1:
        raise ValueError("--repeats and --turns must be positive")
    conditions = selected_conditions(args.conditions)
    models = Models.from_env(mode=args.model_mode)
    preflight = CompletionClient(models, args.output_dir / "preflight_trace.jsonl")
    preflight.preflight()
    if args.check:
        print(f"All required {args.model_mode} models are available.")
        return

    source_case = load_patient_case("no_complaint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    design = {"case": "no_complaint", "conditions": conditions,
              "repeats": args.repeats, "profiles_per_graph": 2, "turns": args.turns,
              "frequency": args.frequency, "orientation": args.orientation,
              "models": json.loads(json.dumps(models.__dict__)),
              "graph_model_endpoint": (
                  f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').rstrip('/')}/v1"
                  if args.model_mode == "local" else
                  (os.getenv("GRAPH_MODEL_BASE_URL") or "http://localhost:11434/v1")
              ),
              "dialogue_endpoint": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
              "sources": ["https://osf.io/y3ta6/",
                          "https://arxiv.org/pdf/2512.05836",
                          "https://github.com/UKPLab/emnlp2026-graph2counsel"]}
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if {k: previous[k] for k in design} != design:
            raise ValueError("Output directory already belongs to a different design")
    else:
        write_json(manifest_path, {**design, "created_at": datetime.now(UTC).isoformat()})

    for condition in conditions:
        symptom = None if condition == "no_symptoms" else condition
        for repeat in range(1, args.repeats + 1):
            run_dir = args.output_dir / condition / f"run_{repeat:02d}"
            client = CompletionClient(models, run_dir / "llm_trace.jsonl")
            graph_path = run_dir / "source_graph.json"
            if graph_path.exists():
                graph = read_json(graph_path)
                if not (run_dir / "source_graph.svg").exists():
                    render_svg(graph, run_dir / "source_graph.svg")
            else:
                units = source_units(source_case, symptom, args.frequency)
                write_json(run_dir / "source_input.json", {"case": source_case,
                           "symptom": symptom, "frequency": args.frequency,
                           "units": units})
                graph = construct_graph(client, units,
                    json.dumps(source_case, ensure_ascii=False), source_kind="prior_plus_phq9")
                write_json(graph_path, graph)
                render_svg(graph, run_dir / "source_graph.svg")
            profiles_path = run_dir / "profiles.json"
            if profiles_path.exists():
                profiles = read_json(profiles_path)
            else:
                profiles = generate_profiles(client, graph, source_case, symptom, args.frequency)
                write_json(profiles_path, profiles)
            if len(profiles) != 2:
                raise ValueError(f"Wrong number of profiles in {profiles_path}")

            for number, profile in enumerate(profiles, start=1):
                folder = run_dir / f"profile_{number:02d}"
                write_json(folder / "profile.json", profile)
                turns = _run_dialogue(profile, graph_path, condition, repeat, number,
                                      folder, models, args.orientation, args.turns,
                                      args.max_tokens)
                reverse_path = folder / "reconstructed_graph.json"
                if reverse_path.exists():
                    reverse_graph = read_json(reverse_path)
                    if not (folder / "reconstructed_graph.svg").exists():
                        render_svg(reverse_graph, folder / "reconstructed_graph.svg")
                else:
                    reverse_graph = construct_graph(client, transcript_units(turns),
                        json.dumps(turns, ensure_ascii=False), source_kind="synthetic_transcript")
                    write_json(reverse_path, reverse_graph)
                    render_svg(reverse_graph, folder / "reconstructed_graph.svg")
                reconstructed_path = folder / "reconstructed_profile.json"
                if reconstructed_path.exists():
                    reconstructed = read_json(reconstructed_path)
                else:
                    reconstructed = reconstruct_profile(client, reverse_graph, turns)
                    write_json(reconstructed_path, reconstructed)
                comparison_path = folder / "comparison.json"
                if not comparison_path.exists():
                    result = compare_round_trip(client, graph, profile, reverse_graph,
                                                reconstructed, symptom)
                    write_json(comparison_path, result)
                _summary(args.output_dir)
                print(f"Completed {condition} run {repeat} profile {number}", flush=True)


if __name__ == "__main__":
    main()
