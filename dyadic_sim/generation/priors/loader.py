"""
dyadic_sim/generation/priors/loader.py

Loads and validates YAML prior files from config/priors/.
Returns raw dicts. The dataclasses in therapist_prior.py and
patient_prior.py handle structure and prompt construction.
"""

from pathlib import Path

import yaml

from dyadic_sim.core import paths


def _prior(*parts: str) -> Path:
    """A prior file: the user's config/priors/ first, the bundled default second."""
    return Path(paths.config_path("priors", *parts))


def _yaml_files(*parts: str) -> list[Path]:
    """YAML files in a priors directory; a user file hides a bundled one of the same name."""
    seen: dict[str, Path] = {}
    for directory in paths.config_dirs("priors", *parts):
        for p in sorted(Path(directory).glob("*.yaml")):
            seen.setdefault(p.stem, p)
    return list(seen.values())


def load_therapist_base() -> dict:
    """Load the therapist base priors (always loaded)."""
    path = _prior("therapist", "base.yaml")
    return _load(path)


def load_therapist_variant(orientation: str) -> dict:
    """
    Load an orientation-specific variant, layered on top of base.

    Args:
        orientation: "psychodynamic", "person_centred", or "cbt"
    """
    path = _prior("therapist", "variants", f"{orientation}.yaml")
    if not path.exists():
        available = [p.stem for p in _yaml_files("therapist", "variants")]
        raise FileNotFoundError(
            f"Orientation '{orientation}' not found. "
            f"Available: {available}"
        )
    return _load(path)


def load_symptom_facts() -> dict:
    """What the patient is told about injected and absent symptoms."""
    return _load(_prior("patient", "symptom_facts.yaml"))


def load_patient_case(case_name: str) -> dict:
    """
    Load a patient case file.

    Args:
        case_name: filename without .yaml, e.g. "only_love_can_save_me"
    """
    path = _prior("patient", "cases", f"{case_name}.yaml")
    if not path.exists():
        available = [
            p.stem for p in _yaml_files("patient", "cases")
            if not p.stem.startswith("_")
        ]
        raise FileNotFoundError(
            f"Patient case '{case_name}' not found. "
            f"Available: {available}"
        )
    return _load(path)


def list_cases() -> list[str]:
    """Return all available patient case names."""
    return [
        p.stem
        for p in _yaml_files("patient", "cases")
        if not p.stem.startswith("_")
    ]


def list_orientations() -> list[str]:
    """Return all available therapist orientation variants."""
    return [
        p.stem
        for p in _yaml_files("therapist", "variants")
    ]


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Prior file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        # Malformed file, not a bad argument, so ValueError is right here.
        raise ValueError(f"Prior file must be a YAML mapping: {path}")  # noqa: TRY004
    return data
