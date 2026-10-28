from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml

from symptom_scoring.prior_vocabulary import KEYS

CASES_DIR = Path("config/priors/patient/cases")
BASE_CASE_NAME = "empty_and_invisible"

# Derived, not restated: order and membership follow the prior vocabulary.
SYMPTOM_KEYS = list(KEYS)


def load_case(case_name: str) -> dict:
    path = CASES_DIR / f"{case_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Case not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        # Malformed file, not a bad argument, so ValueError is right here.
        raise ValueError(f"Case file must contain a YAML mapping: {path}")  # noqa: TRY004

    return data


def write_case(case_name: str, data: dict) -> Path:
    path = CASES_DIR / f"{case_name}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return path


def build_no_symptoms_case(base_case: dict) -> dict:
    case_data = copy.deepcopy(base_case)
    case_data.pop("symptoms", None)
    return case_data


def build_single_symptom_case(
    base_case: dict,
    active_symptom: str,
    active_frequency: str = "nearly every day",
    injection: str = "all_nine",
) -> dict:
    """Write the symptom block the patient prior renders into the prompt.

    Two schemes, differing only in whether the eight inactive symptoms appear:

    ``all_nine`` (default)
        All nine PHQ-9 items, eight of them reading "not at all". The eight
        still *name* their symptom in the prompt ("Thoughts that you would be
        better off dead: not at all"), which may prime the patient to touch on
        them. Listing them is intended to discourage the model from inventing
        levels for domains the prompt leaves unnamed; that effect has not been
        measured.

    ``active_only``
        Only the active symptom's line, so no other symptom is named.
    """
    if injection not in {"all_nine", "active_only"}:
        raise ValueError(f"Unknown injection scheme: {injection!r}")

    case_data = copy.deepcopy(base_case)
    if injection == "active_only":
        case_data["symptoms"] = {active_symptom: active_frequency}
    else:
        case_data["symptoms"] = {
            key: (active_frequency if key == active_symptom else "not at all")
            for key in SYMPTOM_KEYS
        }
    return case_data


def run_session(
    therapist: str,
    patient: str,
    case_name: str,
    orientation: str,
    turns: int,
    max_tokens: int,
) -> None:
    cmd = [
        sys.executable,
        "run.py",
        "--therapist",
        therapist,
        "--patient",
        patient,
        "--case",
        case_name,
        "--orientation",
        orientation,
        "--turns",
        str(turns),
        "--max-tokens",
        str(max_tokens),
    ]

    print(f"\nRunning case: {case_name}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run symptom-injection sessions for one patient case: per run, "
            "one control and one session per symptom."
        )
    )
    parser.add_argument("--therapist", required=True, help="Therapist model name")
    parser.add_argument("--patient", required=True, help="Patient model name")
    parser.add_argument(
        "--case",
        default=BASE_CASE_NAME,
        help=f"Base patient case to inject symptoms into (default: {BASE_CASE_NAME})",
    )
    parser.add_argument(
        "--symptoms",
        default=None,
        help=(
            "Comma-separated subset of symptom keys to run (default: all 9). "
            f"Valid: {','.join(SYMPTOM_KEYS)}"
        ),
    )
    parser.add_argument(
        "--frequency",
        default="nearly every day",
        choices=["several days", "more than half the days", "nearly every day"],
        help="PHQ-9 frequency anchor for the active symptom (default: nearly every day)",
    )
    parser.add_argument(
        "--injection",
        choices=["all_nine", "active_only"],
        default="all_nine",
        help=(
            "How the symptom block is written. 'all_nine' (default) lists all 9 "
            "PHQ-9 items with 8 reading 'not at all'; 'active_only' lists just "
            "the active symptom. Affects generation, so the two are "
            "separate experimental arms; never mix them in one prefix."
        ),
    )
    parser.add_argument("--orientation", default="cbt", help="Therapist orientation")
    parser.add_argument("--turns", type=int, default=10, help="Turns per run")
    parser.add_argument("--max-tokens", type=int, default=300, help="Max tokens per response")
    parser.add_argument("--repeats", type=int, default=5, help="How many runs per condition")
    parser.add_argument(
        "--start-run",
        type=int,
        default=1,
        help=(
            "First run number to use. Defaults to 1. Set it when topping up an "
            "existing cell so the new sessions continue the numbering instead "
            "of colliding: a cell already at n=5 tops up with "
            "--start-run 6 --repeats 5, giving run_6..run_10. Check current "
            "coverage with analysis/validation/design_status.py."
        ),
    )
    parser.add_argument(
        "--prefix",
        default="batch",
        help="Prefix for generated temporary case names",
    )
    parser.add_argument(
        "--keep-generated-cases",
        action="store_true",
        help="Keep generated YAML files instead of deleting them afterward",
    )
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="Skip the no_symptoms runs. Every invocation otherwise generates one "
             "control per run number before the symptom runs, so topping up a "
             "single cell silently regenerates controls at run numbers that "
             "already exist. Use this when filling a gap in an existing design; "
             "check coverage first with analysis/validation/design_status.py.",
    )

    parser.add_argument(
        "--control-only",
        action="store_true",
        help="Generate ONLY the no_symptoms runs, no injected arms. The mirror of "
             "--no-control. Use it to validate a new patient case before "
             "committing a grid to it: generate its control arm alone and read "
             "the base rate of every symptom. A case that produces a symptom "
             "with nothing injected has no headroom for that symptom, and it is "
             "cheaper to find that out from one arm than from ten.",
    )

    args = parser.parse_args()

    if args.control_only and args.no_control:
        parser.error("--control-only and --no-control together would generate nothing")

    if args.control_only:
        symptom_keys = []
    elif args.symptoms is not None and not args.symptoms.strip():
        # Explicitly empty is almost certainly an attempt to say "no symptoms",
        # and it used to fall through to the default of ALL NINE - turning an
        # intended 100-session control arm into a 1000-session grid. Say so.
        parser.error(
            "--symptoms was given but empty. For the control arm alone use "
            "--control-only; omitting --symptoms runs all nine."
        )
    elif args.symptoms:
        symptom_keys = [s.strip() for s in args.symptoms.split(",") if s.strip()]
        invalid = [s for s in symptom_keys if s not in SYMPTOM_KEYS]
        if invalid:
            raise ValueError(f"Unknown symptom key(s): {invalid}. Valid: {SYMPTOM_KEYS}")
    else:
        symptom_keys = SYMPTOM_KEYS

    base_case = load_case(args.case)
    generated_case_paths: list[Path] = []

    try:
        run_numbers = range(args.start_run, args.start_run + args.repeats)

        control_runs = () if args.no_control else run_numbers
        for i in control_runs:
            case_name = f"{args.prefix}_{args.case}_no_symptoms_run_{i}"
            case_data = build_no_symptoms_case(base_case)
            generated_case_paths.append(write_case(case_name, case_data))

            run_session(
                therapist=args.therapist,
                patient=args.patient,
                case_name=case_name,
                orientation=args.orientation,
                turns=args.turns,
                max_tokens=args.max_tokens,
            )

        for symptom_key in symptom_keys:
            for i in run_numbers:
                case_name = f"{args.prefix}_{args.case}_{symptom_key}_run_{i}"
                case_data = build_single_symptom_case(
                    base_case, symptom_key, args.frequency, injection=args.injection
                )
                generated_case_paths.append(write_case(case_name, case_data))

                run_session(
                    therapist=args.therapist,
                    patient=args.patient,
                    case_name=case_name,
                    orientation=args.orientation,
                    turns=args.turns,
                    max_tokens=args.max_tokens,
                )

        print("\nAll runs completed successfully.")
        print("Sessions were saved automatically under data/sessions/.")

    finally:
        if not args.keep_generated_cases:
            for path in generated_case_paths:
                if path.exists():
                    path.unlink()
            print("Temporary generated case files deleted.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nA run failed with exit code {e.returncode}.", file=sys.stderr)
        sys.exit(e.returncode)
    except Exception as e:  # noqa: BLE001 - CLI entry point: report and exit, never traceback
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
