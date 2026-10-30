"""
run.py: dyadic-sim entry point

Usage:
  uv run python run.py --therapist llama3.1 --patient llama3.1 --case afraid_of_dogs
  uv run python run.py --therapist mistral-nemo --patient llama3.1 --case only_love_can_save_me --turns 5
  uv run python run.py --resume data/sessions/session_20260101_120000_abc123   # the turns left, up to turn 10
  uv run python run.py --list-cases
  uv run python run.py --list-orientations
"""

import argparse
import sys

import torch

print("CUDA available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)


from dyadic_sim.core import paths

paths.load_env()

from dyadic_sim.cli.sampling import add_temperature_arguments, temperature_kwargs
from dyadic_sim.generation.dyad import Dyad, session_length
from dyadic_sim.generation.priors.loader import list_cases, list_orientations
from dyadic_sim.generation.session import IncompatibleSessionError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="dyadic-sim: run a therapist-patient agent simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Main mode
    parser.add_argument("--therapist", type=str, help="Therapist model name (e.g. llama3.1)")
    parser.add_argument("--patient", type=str, help="Patient model name (e.g. llama3.1)")
    parser.add_argument("--case", type=str, default="afraid_of_dogs", help="Patient case name")
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Optional synthetic patient identifier saved in session metadata",
    )
    parser.add_argument(
        "--orientation",
        type=str,
        default="cbt",
        choices=["cbt"],
        help="Therapist orientation variant",
    )
    parser.add_argument(
        "--turns", type=int, default=10,
        help="Number of turns to run (default %(default)s; the intake ends at turn 10)",
    )
    parser.add_argument("--max-tokens", type=int, default=300, help="Max tokens per response")
    add_temperature_arguments(parser)

    # Resume mode
    parser.add_argument("--resume", type=str, help="Resume a session by its directory path")
    parser.add_argument(
        "--additional-turns",
        type=int,
        default=None,
        help="How many more turns to run when resuming (default: the turns left "
             "before the intake ends at turn 10; a complete session is not continued)",
    )

    # Info modes
    parser.add_argument("--list-cases", action="store_true", help="List available patient cases")
    parser.add_argument(
        "--list-orientations", action="store_true", help="List available therapist orientations"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # -- Info modes ----------------------------------------------------------

    if args.list_cases:
        cases = list_cases()
        print("\nAvailable patient cases:")
        for c in sorted(cases):
            print(f"  {c}")
        print()
        return

    if args.list_orientations:
        orientations = list_orientations()
        print("\nAvailable therapist orientations:")
        for o in sorted(orientations):
            print(f"  {o}")
        print()
        return

    # -- Resume mode ---------------------------------------------------------

    if args.resume:
        session_id = args.resume.rstrip("/").split("/")[-1]
        print(f"\nResuming session: {session_id}")

        # Load metadata to recover model names
        import json
        from pathlib import Path
        meta_path = Path(args.resume) / "metadata.json"
        if not meta_path.exists():
            print(f"Error: metadata.json not found in {args.resume}")
            sys.exit(1)
        with open(meta_path) as f:
            meta = json.load(f)

        try:
            dyad = Dyad(
                therapist_model=meta["therapist_model"],
                patient_model=meta["patient_model"],
                case_name=meta["case_name"],
                orientation=meta.get("orientation", "psychodynamic"),
                session_id=session_id,
                patient_id=args.patient_id,
            )
            dyad.run(n_turns=args.additional_turns, max_tokens=args.max_tokens,
                     **temperature_kwargs(args))
        except IncompatibleSessionError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        return

    # -- Standard session ----------------------------------------------------

    therapist = args.therapist
    patient = args.patient

    if not therapist or not patient:
        print(
            "\nError: --therapist and --patient are required.\n"
            "Example: uv run python run.py --therapist llama3.1 --patient llama3.1\n"
            "\nFor a list of available models, check config/models.yaml\n"
            "For available cases:       uv run python run.py --list-cases\n"
            "For available orientations: uv run python run.py --list-orientations\n"
        )
        sys.exit(1)

    length = session_length(args.orientation)
    if length is not None and args.turns > length:
        print(f"Error: the intake ends at turn {length}; --turns {args.turns} asks for more.")
        sys.exit(1)

    dyad = Dyad(
        therapist_model=therapist,
        patient_model=patient,
        case_name=args.case,
        orientation=args.orientation,
        patient_id=args.patient_id,
    )

    session = dyad.run(n_turns=args.turns, max_tokens=args.max_tokens,
                       **temperature_kwargs(args))

    print(f"\nSession saved: data/sessions/{session.session_id}/")
    print(f"Transcript:    data/sessions/{session.session_id}/transcript.jsonl")
    print()


if __name__ == "__main__":
    main()
