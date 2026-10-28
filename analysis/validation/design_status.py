"""Which design cells have been run, at what n, and what is missing.

Derived from the sessions on disk rather than a hand-maintained ledger, so it
cannot drift out of sync with reality. A cell is (prefix, case, symptom); n is
the number of sessions in it.

    PYTHONPATH=. python analysis/validation/design_status.py --prefix <prefix>

Top up a short cell by continuing the run numbering rather than restarting it:

    ... --prefix <prefix> --case <case> --symptoms sleep_problems \\
        --repeats 5 --start-run 6

Sessions are counted, not run numbers: a crashed batch that left run_3 twice
still reports the true session count. `next run` is the number to pass to
--start-run, one past the highest run number seen, so a top-up never collides
even when numbering has gaps.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

from analysis.validation.design import check, format_report
from analysis.validation.paths import SESSIONS
from analysis.validation.sessions import parse_case, scan_sessions


def _turns_written(directory: str) -> int:
    """Turns on disk so far, for reporting how far a fragment got."""
    path = os.path.join(directory, "transcript.jsonl")
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def collect(session_glob: str) -> tuple[dict, list[tuple[str, str, int]]]:
    """Map (prefix, case, symptom) -> {n, max_run, generation_versions}.

    Returns the cells, a list of incomplete sessions, and a map of
    (prefix, case, symptom, run) -> directories for duplicate detection. `n` counts COMPLETE
    sessions only. `turn_count` is written as 0 when a session starts and set to
    the real total when it finishes, so it is 0 exactly while a run is in flight
    or was killed partway. Counting those as done is not a cosmetic problem: it
    reports a cell as full when it is short, and it silently mixes 5-turn
    fragments into every downstream analysis.
    """
    cells: dict = defaultdict(
        lambda: {"n": 0, "max_run": 0, "generation_versions": set()}
    )
    partial: list[tuple[str, str, int]] = []
    duplicates: dict = defaultdict(list)
    for directory in sorted(glob.glob(session_glob)):
        meta_path = os.path.join(directory, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        prefix, case, symptom, run = parse_case(meta.get("case_name", ""))
        if not (prefix and case and symptom):
            continue
        cell = cells[(prefix, case, symptom)]
        # Reserve the run number either way, so a top-up cannot collide with a
        # session that is still being written.
        cell["max_run"] = max(cell["max_run"], run or 0)
        if not meta.get("turn_count", 0):
            partial.append((directory, meta.get("case_name", "?"), _turns_written(directory)))
            continue
        cell["n"] += 1
        duplicates[(prefix, case, symptom, run)].append(directory)
        # Absent means generation_version 1 (pilot); see simulation/utterance.py.
        cell["generation_versions"].add(meta.get("generation_version", 1))
    return cells, partial, duplicates


def report(cells: dict, partial: list, duplicates: dict,
           prefix_filter: str | None, target: int | None) -> None:
    keys = sorted(k for k in cells if prefix_filter in (None, k[0]))
    if not keys:
        print(f"No sessions found{f' for prefix {prefix_filter!r}' if prefix_filter else ''}.")
        return

    prefixes = sorted({k[0] for k in keys})
    for prefix in prefixes:
        rows = [k for k in keys if k[0] == prefix]
        cases = sorted({k[1] for k in rows})
        symptoms = sorted({k[2] for k in rows})
        print(f"\n=== prefix: {prefix} ===")
        header = f"{'symptom':34s}" + "".join(f"{c[:20]:>22s}" for c in cases)
        print(header)
        print("-" * len(header))
        for symptom in symptoms:
            line = f"{symptom:34s}"
            for case in cases:
                cell = cells.get((prefix, case, symptom))
                if not cell:
                    line += f"{'.':>22s}"
                    continue
                mark = ""
                if target is not None and cell["n"] < target:
                    mark = f" (+{target - cell['n']})"
                versions = cell["generation_versions"]
                # Empty when every session in the cell is still incomplete;
                # there is no version to report, so say nothing.
                vtag = ("" if versions in ({3}, set())
                        else f" v{','.join(str(v) for v in sorted(versions))}")
                line += f"{f'n={cell[chr(110)]}{mark}{vtag}':>22s}"
            print(line)

        total = sum(cells[k]["n"] for k in rows)
        next_run = max(cells[k]["max_run"] for k in rows) + 1
        print(f"\n  sessions: {total}   next run number for a top-up: --start-run {next_run}")
        if target is not None:
            missing = sum(
                max(0, target - cells[(prefix, case, symptom)]["n"])
                if (prefix, case, symptom) in cells
                else target
                for case in cases
                for symptom in symptoms
            )
            if missing:
                print(f"  sessions still needed to reach n={target} in every listed cell: {missing}")
            else:
                print(f"  every listed cell is at n>={target}")

        # Duplicate (case, condition, run) keys mean two session directories
        # claim the same cell, which double-counts it in every analysis. Distinct
        # from a partial: both may be complete.
        dupes = {k: v for k, v in duplicates.items() if k[0] == prefix and len(v) > 1}
        if dupes:
            print(f"\n  DUPLICATE (case, condition, run) keys ({len(dupes)}):")
            for (_, case, sym, run), dirs in sorted(dupes.items()):
                print(f"    {case}/{sym}/run_{run}: {len(dirs)} directories")
                for d in dirs:
                    print(f"        {d}")

        # Not counted above. Either a run is in flight right now, or one was
        # killed and left a fragment that every analysis would otherwise read.
        stale = [row for row in partial if row[1].startswith(f"{prefix}_")]
        if stale:
            print(f"\n  INCOMPLETE, not counted ({len(stale)}):")
            for directory, case_name, turns in stale:
                print(f"    {case_name}  {turns} turns  {directory}")
            print("    If no run is active, move these out of data/sessions/ before analysing.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", help="Only report this run prefix")
    parser.add_argument(
        "--target",
        type=int,
        help="Flag cells below this n and total how many sessions are missing",
    )
    parser.add_argument(
        "--sessions",
        default=os.path.join(SESSIONS, "session_*"),
        help="Glob for session directories",
    )
    args = parser.parse_args()
    cells, partial, duplicates = collect(args.sessions)
    report(cells, partial, duplicates, args.prefix, args.target)

    # Against the DECLARED design, not merely against itself. Counting sessions
    # says how many exist; only `config/designs.yaml` says how many should, and
    # whether a cell that exists was ever meant to.
    entries = scan_sessions(args.sessions)
    prefixes = ([args.prefix] if args.prefix
                else sorted({e["prefix"] for e in entries if e["prefix"]}))
    print()
    for prefix in prefixes:
        print(format_report(check(prefix, entries)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
