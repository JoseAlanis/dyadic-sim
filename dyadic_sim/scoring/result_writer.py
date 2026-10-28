"""Write auditable JSON and flat turn-level CSV results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_session_json(result: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(output)
    return output


def write_turn_csv(result: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = result.get("turn_results", [])
    fields = [
        "session_id",
        "turn_index",
        "symptom",
        "topic",
        "relevant",
        "accepted_for_scoring",
        "relevance_confidence",
        "relevance_method",
        "evidence",
        "raw_model_output",
        "raw_score",
        "rounded_score",
        "rejection_reason",
        "therapist_text",
        "patient_text",
        "translated_therapist_text",
        "translated_patient_text",
        "context_source",
        "context_synthesized",
        "input_compacted",
    ]
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({"session_id": result.get("session_id"), **row})
    temporary.replace(output)
    return output


def write_directory_summary(results: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["session_id", "patient_id", "topic", "raw_score", "rounded_score", "source_turn"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for topic, aggregate in result["topics"].items():
                writer.writerow(
                    {
                        "session_id": result.get("session_id"),
                        "patient_id": result.get("patient_id"),
                        "topic": topic,
                        "raw_score": aggregate.get("raw_session_score"),
                        "rounded_score": aggregate.get("rounded_session_score"),
                        "source_turn": aggregate.get("source_turn"),
                    }
                )
    return output

