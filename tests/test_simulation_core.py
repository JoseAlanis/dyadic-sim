"""The generator must still read records and case files written by earlier versions.

Older transcript rows and metadata carry fields the current generator no longer
writes (a hidden agenda, a hazard monitor, state snapshots), and older case
files carry blocks it no longer uses. Three things must hold: old transcripts
replay, old case files load without those blocks reaching the prompt, and new
metadata carries the current version and none of the retired keys.

    uv run pytest tests/test_simulation_core.py -q
"""

from __future__ import annotations

import json

import pytest

import dyadic_sim.generation.session as session_mod
from dyadic_sim.generation.priors import patient_prior
from dyadic_sim.generation.session import TurnRecord, new_session, resume_session
from dyadic_sim.generation.utterance import GENERATION_VERSION

# A transcript row exactly as generation_version 3 wrote it, including the two
# fields version 4 no longer has.
V3_ROW = {
    "turn": 1,
    "therapist_text": "How have you been?",
    "patient_text": "Tired, mostly.",
    "therapist_model": "m",
    "patient_model": "m",
    "therapist_tokens": 5,
    "patient_tokens": 3,
    "unconscious_revealed": False,
    "symptom_discussion_started": True,
    "therapist_text_raw": None,
    "patient_text_raw": "(sighs) Tired, mostly.",
    "hazard_flags": ["crisis"],
    "timestamp": "2026-08-12T16:46:19",
}

V3_META = {
    "metadata_schema_version": 2,
    "generation_version": 3,
    "hazard_monitor": "observe_only",
    "session_id": "session_v3",
    "patient_id": None,
    "therapist_model": "m",
    "patient_model": "m",
    "case_name": "afraid_of_dogs_run_1",
    "orientation": "cbt",
    "patient_symptoms": "",
    "patient_symptom_levels": {"sleep_problems": "nearly every day"},
    "conversation_language": "en",
    "initial_patient_prompt": "Hi, please, come on in",
    "turn_count": 1,
    "created_at": "2026-08-12T16:46:19",
    "paused_at_turn": 1,
    "reason": "crisis",
}


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)
    return tmp_path


def test_v3_transcript_replays_without_the_removed_fields(sessions_dir):
    d = sessions_dir / "session_v3"
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps(V3_META))
    (d / "transcript.jsonl").write_text(json.dumps(V3_ROW) + "\n")

    s = resume_session("session_v3")

    assert s.turn_count == 1
    record = s.transcript[0]
    assert record.patient_text == "Tired, mostly."
    assert record.patient_text_raw == "(sighs) Tired, mostly."
    assert "unconscious_revealed" not in TurnRecord.__dataclass_fields__
    assert "hazard_flags" not in TurnRecord.__dataclass_fields__
    assert not hasattr(record, "unconscious_revealed")
    assert not hasattr(record, "hazard_flags")


def test_old_case_file_loads_and_hidden_blocks_never_reach_the_prompt(monkeypatch):
    case = {
        "presenting_complaint": "I'm terrified of dogs.",
        "theory_of_cure": "Exposure, probably.",
        "relational_pattern": "I get on with people.",
        "transference_expectation": "You'll know what to do.",
        "resistance_structure": "I put things off.",
        "hazard_profile": {"to_frame": "low", "notes": "HAZARD-NOTE-SENTINEL"},
        "unconscious_agenda": {
            "content": "AGENDA-SENTINEL",
            "reveal_trigger": "x",
            "reveal_turn_minimum": 1,
        },
    }
    monkeypatch.setattr(patient_prior, "load_patient_case", lambda name: case)

    prior = patient_prior.build_patient_prior("legacy_case")
    prompt = prior.build_system_prompt()

    assert "I'm terrified of dogs." in prompt
    assert "AGENDA-SENTINEL" not in prompt
    assert "HAZARD-NOTE-SENTINEL" not in prompt
    assert not hasattr(prior, "unconscious_agenda")
    assert not hasattr(prior, "hazard_profile")


def test_new_session_metadata_carries_current_version_without_monitor_keys(sessions_dir):
    s = new_session("m", "m", "afraid_of_dogs", "cbt")
    meta = json.loads((s.session_dir / "metadata.json").read_text())

    assert GENERATION_VERSION >= 4  # 4 is where the monitor and agenda were removed
    assert meta["generation_version"] == GENERATION_VERSION
    assert "hazard_monitor" not in meta
    assert "hazard_summary" not in meta
    assert not list(s.session_dir.glob("*_state_snapshots.json"))
