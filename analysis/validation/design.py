"""Check a corpus against the design it is supposed to instantiate.

`config/designs.yaml` states, prospectively, what each arm should contain. This
module compares that to what is on disk and reports three things separately:

  missing      expected cells with fewer runs than declared
  unexpected   sessions present under a prefix the manifest does not describe
  extra        valid runs beyond `runs`, kept or not according to `inclusion`

`runs` is one integer for every cell, or a mapping case -> integer when an
arm's beds are sized differently (say 40 runs for one case and 10 for another).
A case absent from the mapping is an error, not a default.

The distinction matters. Suppose a cell declared for five runs turns out to hold
six. The objection is not that an extra control biases anything - equal-case
weighting absorbs it - but that if no rule said in advance whether it belonged,
any rule chosen now is chosen with knowledge of its effect.

Silence is the failure mode this guards against: a corpus that quietly grew, or
quietly lost a cell, looks identical to one that did neither.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict

import yaml

MANIFEST_PATH = os.path.join("config", "designs.yaml")


def load_manifest(path: str = MANIFEST_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def expected_runs(spec: dict, case: str, prefix: str = "?") -> int:
    """Declared runs for one case; see the module docstring."""
    runs = spec["runs"]
    if isinstance(runs, dict):
        if case not in runs:
            raise KeyError(
                f"{MANIFEST_PATH}: `runs` for {prefix!r} is per-case but has no "
                f"entry for {case!r}"
            )
        return int(runs[case])
    return int(runs)


def check(prefix: str, entries: list[dict], manifest: dict | None = None) -> dict:
    """Compare `scan_sessions()` output for one prefix against its design.

    `entries` are the dicts returned by `sessions.scan_sessions`, unfiltered, so
    that excluded sessions can be reported rather than vanishing.
    """
    manifest = load_manifest() if manifest is None else manifest
    spec = manifest.get(prefix)
    mine = [e for e in entries if e.get("prefix") == prefix]
    if spec is None:
        return {
            "prefix": prefix, "described": False,
            "n_sessions": len(mine),
            "problems": [(f"no entry for {prefix!r} in {MANIFEST_PATH}; "
                          f"{len(mine)} sessions are undescribed")],
            "missing": {}, "extra": {}, "unexpected": [], "by_status": {},
        }

    cases, conditions = set(spec["cases"]), set(spec["conditions"])
    inclusion = spec.get("inclusion", "all_valid")

    counts: dict[tuple[str, str], list[int]] = defaultdict(list)
    unexpected, by_status = [], Counter()
    for e in mine:
        by_status[e["status"]] += 1
        if e["status"] in ("malformed",):
            continue
        cell = (e["case"], e["symptom"])
        if e["case"] not in cases or e["symptom"] not in conditions:
            unexpected.append({"case": e["case"], "condition": e["symptom"],
                               "run": e["run"], "dir": e["dir"],
                               "why": "cell not in the declared design"})
            continue
        counts[cell].append(e["run"])

    missing, extra, declared = {}, {}, {}
    for case in sorted(cases):
        want = expected_runs(spec, case, prefix)
        for condition in sorted(conditions):
            cell = f"{case}/{condition}"
            declared[cell] = want
            runs = sorted(r for r in counts.get((case, condition), []) if r)
            if len(runs) < want:
                missing[cell] = want - len(runs)
            beyond = [r for r in runs if r > want]
            if beyond:
                extra[cell] = beyond

    problems = []
    for cell, n in missing.items():
        problems.append(f"{cell}: {n} run(s) short of {declared[cell]}")
    for cell, runs in extra.items():
        verdict = "included (inclusion: all_valid)" if inclusion == "all_valid" \
            else "EXCLUDED (inclusion: first_n)"
        problems.append(f"{cell}: run(s) {runs} beyond {declared[cell]} - {verdict}")
    for u in unexpected:
        problems.append(f"unexpected cell {u['case']}/{u['condition']} run {u['run']}")

    return {
        "prefix": prefix, "described": True, "inclusion": inclusion,
        "n_sessions": len(mine), "by_status": dict(by_status),
        "missing": missing, "extra": extra, "unexpected": unexpected,
        "problems": problems,
    }


def format_report(result: dict) -> str:
    lines = [f"design check: {result['prefix']}"]
    if not result["described"]:
        lines += [f"  {p}" for p in result["problems"]]
        return "\n".join(lines)
    lines.append(f"  sessions: {result['n_sessions']}  "
                 f"status: {result['by_status']}  "
                 f"inclusion: {result['inclusion']}")
    if not result["problems"]:
        lines.append("  matches the declared design")
    else:
        for p in result["problems"]:
            lines.append(f"  - {p}")
    return "\n".join(lines)
