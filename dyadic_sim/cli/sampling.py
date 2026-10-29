"""Sampling flags shared by `run` and `run-symptom-experiments`."""

import argparse

from dyadic_sim.generation.dyad import DEFAULT_TEMPERATURE


def add_temperature_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for both agents (default %(default)s)",
    )
    parser.add_argument(
        "--patient-temperature", type=float, default=None,
        help="The patient's temperature, if it differs from --temperature",
    )
    parser.add_argument(
        "--therapist-temperature", type=float, default=None,
        help="The therapist's temperature, if it differs from --temperature",
    )


def temperature_kwargs(args: argparse.Namespace) -> dict:
    """Keyword arguments for Dyad.run()."""
    return {
        "temperature": args.temperature,
        "patient_temperature": args.patient_temperature,
        "therapist_temperature": args.therapist_temperature,
    }


def temperature_flags(args: argparse.Namespace) -> list[str]:
    """The same settings as command-line flags, written out in full for a child run."""
    flags = ["--temperature", str(args.temperature)]
    if args.patient_temperature is not None:
        flags += ["--patient-temperature", str(args.patient_temperature)]
    if args.therapist_temperature is not None:
        flags += ["--therapist-temperature", str(args.therapist_temperature)]
    return flags
