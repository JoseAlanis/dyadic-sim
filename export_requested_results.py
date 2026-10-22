"""Export the requested simulation batch into a case/condition hierarchy.

The canonical session records remain under ``data/sessions``.  This exporter
copies each matching record into ``result/<case>/<condition>/run_<n>`` and
writes manifests that make the requested cell counts easy to audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from symptom_scoring.prior_vocabulary import KEYS, LABELS


CASES = (
    "no_complaint",
    "empty_and_invisible",
    "afraid_of_dogs",
    "feeling_off",
    "only_love_can_save_me",
)
CONDITIONS = ("no_symptoms", *KEYS)
EXPECTED = {
    (case, condition): 10
    for case in CASES
    for condition in CONDITIONS
}
EXPECTED[("no_complaint", "no_symptoms")] = 30


def parse_case_name(case_name: str, prefix: str) -> tuple[str, str, int] | None:
    start = f"{prefix}_"
    if not case_name.startswith(start):
        return None

    remainder = case_name[len(start):]
    for base_case in CASES:
        case_start = f"{base_case}_"
        if not remainder.startswith(case_start):
            continue
        condition_and_run = remainder[len(case_start):]
        for condition in CONDITIONS:
            marker = f"{condition}_run_"
            if condition_and_run.startswith(marker):
                run_number = int(condition_and_run[len(marker):])
                return base_case, condition, run_number
    return None


def export(prefix: str, sessions_dir: Path, output_dir: Path) -> None:
    records: list[dict] = []
    seen_cells: set[tuple[str, str, int]] = set()

    for metadata_path in sorted(sessions_dir.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        parsed = parse_case_name(str(metadata.get("case_name", "")), prefix)
        if parsed is None:
            continue

        base_case, condition, run_number = parsed
        cell = (base_case, condition, run_number)
        if cell in seen_cells:
            raise RuntimeError(f"Duplicate requested cell: {cell}")
        seen_cells.add(cell)

        source_dir = metadata_path.parent
        transcript_path = source_dir / "transcript.jsonl"
        pdfs = list(source_dir.glob("*.pdf"))
        if metadata.get("turn_count") != 10:
            raise RuntimeError(f"Expected 10 turns in {source_dir}")
        if not transcript_path.exists() or len(transcript_path.read_text().splitlines()) != 10:
            raise RuntimeError(f"Incomplete transcript in {source_dir}")
        if len(pdfs) != 1:
            raise RuntimeError(f"Expected exactly one PDF in {source_dir}, found {len(pdfs)}")

        target_dir = output_dir / base_case / condition / f"run_{run_number:02d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(transcript_path, target_dir / "transcript.jsonl")
        shutil.copy2(metadata_path, target_dir / "metadata.json")
        shutil.copy2(pdfs[0], target_dir / "transcript.pdf")

        condition_label = "No symptom" if condition == "no_symptoms" else LABELS[condition]
        condition_info = {
            "base_case": base_case,
            "condition": condition,
            "condition_label": condition_label,
            "symptom_frequency": (
                None if condition == "no_symptoms" else metadata["patient_symptom_levels"][condition]
            ),
            "run_number": run_number,
            "session_id": metadata["session_id"],
            "therapist_model": metadata["therapist_model"],
            "patient_model": metadata["patient_model"],
            "turn_count": metadata["turn_count"],
            "generation_version": metadata["generation_version"],
            "source_directory": str(source_dir),
        }
        (target_dir / "condition.json").write_text(
            json.dumps(condition_info, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        records.append(condition_info)

    counts = Counter((r["base_case"], r["condition"]) for r in records)
    missing = {
        f"{case}/{condition}": expected - counts[(case, condition)]
        for (case, condition), expected in EXPECTED.items()
        if counts[(case, condition)] != expected
    }
    if missing:
        raise RuntimeError(f"Requested matrix is incomplete or overfull: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records.sort(key=lambda r: (r["base_case"], r["condition"], r["run_number"]))
    (output_dir / "manifest.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    summary = [
        "# Requested simulation results",
        "",
        f"Total sessions: {len(records)}",
        "",
        "- Therapist model: mistral-nemo",
        "- Patient model: mistral-nemo",
        "- Turns per session: 10",
        "- Symptom injection: active_only",
        "- Active symptom frequency: nearly every day",
        "",
        "## Counts",
        "",
    ]
    for case in CASES:
        summary.append(f"### {case}")
        summary.append("")
        for condition in CONDITIONS:
            summary.append(f"- {condition}: {counts[(case, condition)]}")
        summary.append("")
    (output_dir / "README.md").write_text("\n".join(summary), encoding="utf-8")

    print(f"Exported and verified {len(records)} sessions to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="requested_20260908")
    parser.add_argument("--sessions-dir", type=Path, default=Path("data/sessions"))
    parser.add_argument("--output-dir", type=Path, default=Path("result"))
    args = parser.parse_args()
    export(args.prefix, args.sessions_dir, args.output_dir)


if __name__ == "__main__":
    main()
