"""Frozen, display-ready snapshots of simulated sessions.

A snapshot is exactly the text a rater sees, in chronological order, with
positional utterance ids (`u00`, `u01`, ...). Evidence spans point into it with
zero-based Unicode code-point offsets, the same indexing as Python string
slicing, so `text[start:end] == quote` is the whole integrity check.

**Chronology is specific to this repository.** A `transcript.jsonl` row holds
the PATIENT turn first and then the therapist's reply to it. The line a patient
turn answers is the previous row's `therapist_text`, or, for the first row, the
metadata field `initial_patient_prompt`, which despite its name is the
therapist's opening line (see `dyadic_sim.scoring.transcript_parser.build_turn_pairs`).

Nothing is dropped or rewritten: QC-flagged turns, echo artefacts and the final
therapist reply are all shown, and text is used verbatim from the stored fields.
A missing opener is recorded as missing, never reconstructed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PREPROCESSING = "verbatim-v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_session(directory: Path) -> tuple[dict, list[dict], dict]:
    """(metadata, transcript rows, file hashes) for one session directory."""
    meta_bytes = (directory / "metadata.json").read_bytes()
    tx_bytes = (directory / "transcript.jsonl").read_bytes()
    rows = [json.loads(line) for line in tx_bytes.decode("utf-8").splitlines() if line.strip()]
    hashes = {
        "metadata_sha256": sha256_bytes(meta_bytes),
        "transcript_sha256": sha256_bytes(tx_bytes),
    }
    return json.loads(meta_bytes), rows, hashes


def dialogue(meta: dict, rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Chronological utterances, their sealed sources, and provenance notes.

    Order: [therapist opener], patient 1, therapist 1, ..., patient n, therapist n.
    The first list is what the page shows; the second maps each utterance back
    to the stored field it came from and never leaves the researcher's side.
    """
    shown: list[dict] = []
    sources: list[dict] = []
    notes = {"opener": "missing", "empty_utterances": []}

    def add(speaker: str, text, turn: int, source: dict) -> None:
        uid = f"u{len(shown):02d}"
        text = "" if text is None else str(text)
        shown.append({"id": uid, "speaker": speaker, "turn": turn, "text": text})
        sources.append({"id": uid, **source})
        if not text.strip():
            notes["empty_utterances"].append(uid)

    opener = meta.get("initial_patient_prompt")
    if isinstance(opener, str) and opener.strip():
        notes["opener"] = "recorded"
        add("therapist", opener, 0, {"field": "metadata.initial_patient_prompt"})
    for index, row in enumerate(rows):
        turn = row.get("turn") if isinstance(row.get("turn"), int) else index + 1
        add("patient", row.get("patient_text"), turn, {"row": index, "field": "patient_text"})
        add("therapist", row.get("therapist_text"), turn, {"row": index, "field": "therapist_text"})
    return shown, sources, notes


def digest(utterances: list[dict]) -> str:
    """Content hash of the displayed text, independent of dict ordering."""
    canonical = json.dumps(
        [[u["id"], u["speaker"], u["turn"], u["text"]] for u in utterances],
        ensure_ascii=False, separators=(",", ":"),
    )
    return "sha256:" + sha256_bytes(canonical.encode("utf-8"))


def patient_ids(utterances: list[dict]) -> list[str]:
    return [u["id"] for u in utterances if u["speaker"] == "patient"]
