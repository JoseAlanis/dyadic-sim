"""Draw sessions from any batch into an explicit build manifest.

    uv run python -m study_web.select_sessions --prefix <prefix> \
        --cases no_complaint,empty_and_invisible --runs 1-10 \
        --shared no_complaint=2,empty_and_invisible=1 --unique no_complaint=1 \
        --repeats 3 --raters R1,R2,R3 --seed 1 \
        --study-version <study> \
        --out data/results/clinician_study/manifests/<study>.json

Any batch works. `--prefix` picks the run prefix inside each case_name, as the
analysis CLIs do, and `--sessions` takes the same globs, so an imported batch
under data/external/<name>/ is read like a local one. `--prefix ''` means every
prefix, which pools batches, so it has to be asked for.

Sampling is a simple stratified random draw, one stratum per (case, condition):

  --shared   sessions per cell rated by every rater (agreement, validation)
  --unique   sessions per cell per rater, each rated by one rater only
             (development), disjoint from the shared ones and from each other
  --repeats  shared sessions each rater re-rates at the end (drift)

Counts are one number for every cell, `all`, or per case (`case=N,case=N`); a
case left out of a per-case list gets none. A cell too small for its request is
an error unless --allow-short, so a design is never shrunk silently.

Only complete sessions are eligible. Sessions listed in any --exclude-manifest
are removed first, which keeps demo, calibration and study sets disjoint. The
manifest records the frame size of every cell, what was drawn, the seed and the
command, so inclusion probabilities can be computed from it later. Reading a
batch here opens metadata only, never a transcript.
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from analysis.validation.sessions import COMPLETE, scan_sessions
from study_web import codebook

REPO = Path(__file__).resolve().parents[1]
DEFAULT_GLOB = "data/sessions/session_*"


class SelectionError(RuntimeError):
    """The request cannot be met as stated."""


def parse_runs(spec: str | None) -> set[int] | None:
    """`1-10,15` -> {1, ..., 10, 15}; None means every run."""
    if not spec:
        return None
    runs: set[int] = set()
    for part in spec.split(","):
        low, _, high = part.strip().partition("-")
        try:
            runs |= set(range(int(low), int(high or low) + 1))
        except ValueError as exc:
            raise SelectionError(f"cannot read run range {part!r}") from exc
    return runs


def parse_counts(spec: str | None, cases: list[str]) -> dict[str, int | str]:
    """`2`, `all`, or `case=N,case=N` -> {case: N or 'all'}."""
    if not spec:
        return {case: 0 for case in cases}
    def value(text: str) -> int | str:
        text = text.strip()
        if text == "all":
            return "all"
        if not text.isdigit():
            raise SelectionError(f"count {text!r} must be a whole number or 'all'")
        return int(text)
    if "=" not in spec:
        return {case: value(spec) for case in cases}
    counts = {case: 0 for case in cases}
    for part in spec.split(","):
        case, _, number = part.partition("=")
        if case.strip() not in counts:
            raise SelectionError(f"count given for {case.strip()!r}, which is not a selected case")
        counts[case.strip()] = value(number)
    return counts


def excluded_paths(manifests: list[Path]) -> set[str]:
    paths: set[str] = set()
    for manifest in manifests:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        paths |= {normalise(e["path"]) for e in [*data.get("sessions", []), *data.get("repeats", [])]}
    return paths


def normalise(path: str) -> str:
    """Repo-relative where possible, so the same session always has one spelling."""
    candidate = Path(path)
    candidate = (candidate if candidate.is_absolute() else REPO / candidate).resolve()
    try:
        return str(candidate.relative_to(REPO))
    except ValueError:
        return str(candidate)


def select(args: argparse.Namespace) -> dict:
    raters = [r.strip() for r in args.raters.split(",") if r.strip()]
    if not raters or len(set(raters)) != len(raters):
        raise SelectionError("--raters must list unique pseudonyms, e.g. R1,R2,R3")
    runs = parse_runs(args.runs)
    wanted_cases = [c.strip() for c in args.cases.split(",")] if args.cases else None
    wanted_conditions = [c.strip() for c in args.conditions.split(",")] if args.conditions else None
    if wanted_conditions:
        unknown = set(wanted_conditions) - {"no_symptoms", *[k for k, _, _ in codebook.SYMPTOMS]}
        if unknown:
            raise SelectionError(f"unknown conditions {sorted(unknown)}")
    exclude = excluded_paths(args.exclude_manifest)

    globs = [str(REPO / g) if not Path(g).is_absolute() else g for g in args.sessions]
    frame: dict[tuple[str, str], list[dict]] = defaultdict(list)
    set_aside: Counter = Counter()
    for entry in scan_sessions(globs):
        if args.prefix and entry["prefix"] != args.prefix:
            continue
        if wanted_cases and entry["case"] not in wanted_cases:
            continue
        if wanted_conditions and entry["symptom"] not in wanted_conditions:
            continue
        if runs is not None and entry["run"] not in runs:
            continue
        path = normalise(entry["dir"])
        if entry["status"] != COMPLETE:
            set_aside[f"status {entry['status']}"] += 1
            continue
        if path in exclude:
            set_aside["listed in an --exclude-manifest"] += 1
            continue
        frame[(entry["case"], entry["symptom"])].append({"path": path, "run": entry["run"]})
    if not frame:
        raise SelectionError("no eligible session matches the prefix and filters")

    cases = sorted({case for case, _ in frame})
    shared_n = parse_counts(args.shared, cases)
    unique_n = parse_counts(args.unique, cases)
    if not any(shared_n.values()) and not any(unique_n.values()):
        raise SelectionError("nothing to draw: give --shared and/or --unique")
    if args.repeats and not any(shared_n.values()):
        raise SelectionError("--repeats re-rates shared sessions, so it needs --shared")

    rng = random.Random(f"{args.seed}|{args.study_version}")
    sessions, cells, shortfalls = [], {}, []
    for cell in sorted(frame):
        case, condition = cell
        pool = sorted(frame[cell], key=lambda e: e["path"])
        rng.shuffle(pool)
        size = len(pool)
        n_shared = size if shared_n[case] == "all" else shared_n[case]
        n_unique = (size - n_shared) // len(raters) if unique_n[case] == "all" else unique_n[case]
        need = n_shared + n_unique * len(raters)
        if need > size:
            shortfalls.append(f"{case}/{condition}: needs {need}, has {size}")
            if not args.allow_short:
                continue
            n_shared = min(n_shared, size)
            n_unique = (size - n_shared) // len(raters)
        drawn_shared = pool[:n_shared]
        rest = pool[n_shared:]
        for item in drawn_shared:
            sessions.append({"path": item["path"], "set": "shared", "raters": raters,
                             "cell": {"case": case, "condition": condition, "run": item["run"]}})
        for index, rater in enumerate(raters):
            for item in rest[index * n_unique:(index + 1) * n_unique]:
                sessions.append({"path": item["path"], "set": "unique", "raters": [rater],
                                 "cell": {"case": case, "condition": condition, "run": item["run"]}})
        cells[f"{case}/{condition}"] = {
            "frame": size, "shared": n_shared, "unique_per_rater": n_unique,
            "p_shared": n_shared / size, "p_unique_any_rater": n_unique * len(raters) / size,
        }
    if shortfalls and not args.allow_short:
        raise SelectionError("cells too small for the request: " + "; ".join(shortfalls))

    shared_paths = [s["path"] for s in sessions if s["set"] == "shared"]
    if args.repeats > len(shared_paths):
        raise SelectionError(f"--repeats {args.repeats} exceeds the {len(shared_paths)} shared sessions")
    repeats = []
    for rater in raters:
        chooser = random.Random(f"{args.seed}|{args.study_version}|{rater}|repeats")
        repeats += [{"path": p, "rater": rater, "set": "repeat"} for p in chooser.sample(shared_paths, args.repeats)]

    return {
        "study_version": args.study_version,
        "protocol_version": codebook.PROTOCOL_VERSION,
        "description": args.description,
        "seed": args.seed,
        "raters": raters,
        "sessions": sessions,
        "repeats": repeats,
        "selection": {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": shlex.join(["python", "-m", "study_web.select_sessions", *sys.argv[1:]]),
            "sessions_globs": args.sessions, "prefix": args.prefix,
            "cases": wanted_cases, "conditions": wanted_conditions, "runs": args.runs,
            "design": {"shared": args.shared, "unique": args.unique, "repeats": args.repeats},
            "exclude_manifests": [str(m) for m in args.exclude_manifest],
            "set_aside": dict(set_aside), "shortfalls": shortfalls, "cells": cells,
        },
    }


def summary(manifest: dict) -> str:
    rows = [f"  {'cell':50s} {'frame':>5s} {'shared':>6s} {'unique/rater':>12s}"]
    for cell, info in manifest["selection"]["cells"].items():
        rows.append(f"  {cell:50s} {info['frame']:5d} {info['shared']:6d} {info['unique_per_rater']:12d}")
    per_rater = Counter(r for s in manifest["sessions"] for r in s["raters"])
    for repeat in manifest["repeats"]:
        per_rater[repeat["rater"]] += 1
    rows.append(f"  distinct sessions {len(manifest['sessions'])}; assignments per rater "
                + ", ".join(f"{r} {n}" for r, n in sorted(per_rater.items())))
    for reason, n in manifest["selection"]["set_aside"].items():
        rows.append(f"  set aside: {n} ({reason})")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prefix", required=True, help="Run prefix to draw from; '' for every prefix.")
    parser.add_argument("--sessions", nargs="+", default=[DEFAULT_GLOB], help="Session directory globs.")
    parser.add_argument("--cases", help="Comma-separated cases; default all.")
    parser.add_argument("--conditions", help="Comma-separated conditions incl. no_symptoms; default all.")
    parser.add_argument("--runs", help="Run numbers, e.g. 1-10 or 1-10,15; default all.")
    parser.add_argument("--shared", help="Per cell, rated by all raters: N, all, or case=N,...")
    parser.add_argument("--unique", help="Per cell and rater, rated by one rater: N, all, or case=N,...")
    parser.add_argument("--repeats", type=int, default=0, help="Shared sessions each rater re-rates at the end.")
    parser.add_argument("--raters", required=True, help="Comma-separated pseudonyms, e.g. R1,R2,R3.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--study-version", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--exclude-manifest", nargs="*", type=Path, default=[])
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--out", type=Path, help="Manifest to write; omit to only print the draw.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing --out.")
    args = parser.parse_args()
    try:
        manifest = select(args)
    except SelectionError as exc:
        print(f"selection refused: {exc}", file=sys.stderr)
        return 1
    print(summary(manifest))
    if args.out:
        if args.out.exists() and not args.force:
            print(f"{args.out} exists; pass --force to replace it", file=sys.stderr)
            return 1
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
