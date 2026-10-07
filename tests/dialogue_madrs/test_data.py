from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dialogue_madrs.data import (
    DialogueSample,
    build_dialogue_samples,
    build_synthetic_workbook_samples,
    format_symptom_input,
    grouped_split,
)


def test_formats_symptom_aware_patient_only_input():
    text = format_symptom_input("Reduced Sleep", "I wake up at four.")
    assert text == "SYMPTOM: Reduced Sleep [SEP] I wake up at four."


def test_preprocessing_extracts_only_patient_turns(tmp_path):
    session_dir = tmp_path / "S1"
    session_dir.mkdir()
    transcript = [
        {"therapist_text": "How are you?", "patient_text": "I feel low."},
        {"therapist_text": "How is sleep?", "patient_text": "I wake early."},
    ]
    (session_dir / "transcript.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in transcript), encoding="utf-8"
    )
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "session_id": "S1",
                "patient_id": "P1",
                "symptom": "Reduced Sleep",
                "severity_score": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = build_dialogue_samples(tmp_path, labels)

    assert len(samples) == 1
    assert samples[0].patient_text == "I feel low.\nI wake early."
    assert "How are you?" not in samples[0].patient_text
    assert samples[0].severity_score == 3.0


def test_grouped_split_never_leaks_patients():
    samples = [
        DialogueSample(
            f"S{patient}{session}",
            symptom,
            "patient speech",
            float(session),
            f"P{patient}",
        )
        for patient in range(6)
        for session in range(2)
        for symptom in ("Sleep", "Sadness")
    ]

    splits = grouped_split(samples, seed=7)
    patient_sets = [
        {sample.patient_id for sample in split} for split in splits.values()
    ]
    assert all(patient_sets)
    assert patient_sets[0].isdisjoint(patient_sets[1])
    assert patient_sets[0].isdisjoint(patient_sets[2])
    assert patient_sets[1].isdisjoint(patient_sets[2])


def test_partial_patient_ids_are_rejected():
    samples = [
        DialogueSample("S1", "Sleep", "text", 1, "P1"),
        DialogueSample("S2", "Sleep", "text", 2),
        DialogueSample("S3", "Sleep", "text", 3, "P3"),
    ]
    with pytest.raises(ValueError, match="patient_id"):
        grouped_split(samples)


def test_workbook_reader_uses_patient_item_speech_only():
    rows = iter(
        [
            ("Subject", "Speaker", "Transcription", "Topic", "Score"),
            ("S1", "SPEAKER_00", "How do you sleep?", "Schlaf", 6),
            ("S1", "SPEAKER_01", "I wake early.", "Schlaf", 3),
            ("S1", "SPEAKER_01", "And cannot fall back asleep.", "Schlaf", 3),
            ("S1", "SPEAKER_01", "Generally okay.", "Allgemein", 2),
        ]
    )
    workbook = SimpleNamespace(
        worksheets=[object()],
        active=SimpleNamespace(iter_rows=lambda **kwargs: rows),
        close=lambda: None,
    )
    with patch("openpyxl.load_workbook", return_value=workbook):
        samples = build_synthetic_workbook_samples("source.xlsx")

    assert len(samples) == 1
    assert samples[0].patient_id == "S1"
    assert samples[0].symptom == "Schlaf"
    assert samples[0].patient_text == "I wake early.\nAnd cannot fall back asleep."
    assert samples[0].severity_score == 3
