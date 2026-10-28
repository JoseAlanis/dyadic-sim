"""Load the frozen analysis specification and evaluate results against it.

`config/analysis_spec.yaml` (local; start from `config/analysis_spec.example.yaml`)
fixes the parameters, the estimand, the obligatory sensitivities and what counts
as evidence. This module reads it and applies the criteria, so the verdict on a
cell comes from the declared rule rather than from whichever number looked best.

The criteria are conjunctive on purpose. A symptom can show a larger lexicon lift
than another and still not count, because its detector also rises in arms where
a different symptom was injected, or the embedding does not move. A
magnitude-only rule would admit it; magnitude AND specificity AND agreement in
direction does not.
"""

from __future__ import annotations

import os

import yaml

SPEC_PATH = os.path.join("config", "analysis_spec.yaml")


def load(path: str = SPEC_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Copy config/analysis_spec.example.yaml to "
            f"{SPEC_PATH} and fix it before analysing."
        )
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def evaluate(
    *,
    lexicon_lift_pp: float | None,
    embedding_d: float | None,
    embedding_contrast: float | None,
    max_leak_pp: float | None,
    spec: dict | None = None,
) -> dict:
    """Apply the declared criteria to one (case, symptom) cell.

    `max_leak_pp` is the largest rise of THIS symptom's detector in an arm where
    a different symptom was injected, expressed in percentage points over
    control. None means no other injected arm was available to check, which is
    reported as `unknown` rather than silently passed.
    """
    spec = spec or load()
    c = spec["criteria"]
    checks = {}

    checks["magnitude_lexicon"] = {
        "value": lexicon_lift_pp,
        "threshold": c["magnitude"]["lexicon_lift_pp"],
        "passed": (lexicon_lift_pp is not None
                   and lexicon_lift_pp >= c["magnitude"]["lexicon_lift_pp"]),
    }
    checks["magnitude_embedding"] = {
        "value": embedding_d,
        "threshold": c["magnitude"]["embedding_cohens_d"],
        "passed": (embedding_d is not None
                   and embedding_d >= c["magnitude"]["embedding_cohens_d"]),
    }
    checks["specificity"] = {
        "value": max_leak_pp,
        "threshold": c["specificity"]["max_leak_pp"],
        # Unknown is not a pass. A cell with no comparison arm cannot
        # demonstrate specificity, and saying so is the honest outcome.
        "passed": (max_leak_pp is not None
                   and max_leak_pp <= c["specificity"]["max_leak_pp"]),
        "note": None if max_leak_pp is not None else "no other injected arm to check",
    }
    same_sign = (
        lexicon_lift_pp is not None and embedding_contrast is not None
        and (lexicon_lift_pp > 0) == (embedding_contrast > 0)
    )
    checks["direction"] = {
        "value": same_sign,
        "threshold": c["direction"]["require_same_sign"],
        "passed": bool(same_sign) or not c["direction"]["require_same_sign"],
    }

    failed = [k for k, v in checks.items() if not v["passed"]]
    return {
        "checks": checks,
        "failed": failed,
        "verdict": "supported" if not failed else "not supported",
        "spec_version": spec["version"],
        "spec_status": spec["status"],
    }
