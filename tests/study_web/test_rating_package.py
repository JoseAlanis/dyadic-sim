"""Rating packages: chronology, blinding, and the return-file contract.

Uses the two synthetic sessions in tests/fixtures/study_web, never a real transcript.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from pathlib import Path

import pytest

from dyadic_sim.study_web import RETURN_FORMAT, SCHEMA_VERSION, codebook
from dyadic_sim.study_web.build_package import BuildError, build
from dyadic_sim.study_web.import_returns import run as import_returns
from dyadic_sim.study_web.select_sessions import SelectionError, select
from dyadic_sim.study_web.snapshot import dialogue, read_session

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "study_web"
SLEEP = FIXTURES / "sessions" / "session_20000101_000001_f1a001"
CONTROL = FIXTURES / "sessions" / "session_20000101_000002_f1a002"
KEYS = [key for key, _, _ in codebook.SYMPTOMS]


def payload_of(html: str) -> dict:
    match = re.search(r'<script id="package-data" type="application/json">(.*?)</script>', html, re.DOTALL)
    return json.loads(match.group(1))


@pytest.fixture
def study(tmp_path):
    summary = build(FIXTURES / "manifest.json", tmp_path, now="2000-01-02T00:00:00+00:00")
    return Path(summary["study_dir"])


def package(study: Path, rater: str) -> dict:
    return payload_of((study / "packages" / f"{rater}.html").read_text(encoding="utf-8"))


def test_dialogue_is_chronological_and_complete():
    meta, rows, _ = read_session(SLEEP)
    shown, sources, notes = dialogue(meta, rows)
    # The patient line comes first in each transcript row; the opener precedes it.
    assert [u["speaker"] for u in shown] == ["therapist", "patient", "therapist", "patient", "therapist", "patient", "therapist"]
    assert [u["turn"] for u in shown] == [0, 1, 1, 2, 2, 3, 3]
    assert shown[0]["text"] == meta["initial_patient_prompt"]
    assert shown[-1]["text"] == rows[-1]["therapist_text"], "the final therapist reply must not be dropped"
    assert sources[1] == {"id": "u01", "row": 0, "field": "patient_text"}
    assert notes["opener"] == "recorded"


def test_missing_opener_is_recorded_not_invented():
    meta, rows, _ = read_session(CONTROL)
    shown, _, notes = dialogue(meta, rows)
    assert shown[0]["speaker"] == "patient"
    assert notes["opener"] == "missing"
    assert notes["empty_utterances"] == ["u02"], "an empty patient turn stays visible and is recorded"


def test_packages_carry_no_source_information(study):
    for rater in ("R1", "R2"):
        html = (study / "packages" / f"{rater}.html").read_text(encoding="utf-8")
        for forbidden in ("session_20000101", "fixture01", "no_complaint", "sleep_problems_run", "_run_",
                          "2000-01-01T00:00", "transcript.jsonl", "patient_symptom", "nearly every day; synthetic"):
            assert forbidden not in html, f"{rater}.html contains {forbidden!r}"
        payload = payload_of(html)
        assert set(payload) == {"format", "schema_version", "app_version", "package_id", "study_version",
                                "protocol_version", "rater_id", "built_at", "codebook", "assignments"}
        assert all(set(u) == {"id", "speaker", "turn", "text"} for a in payload["assignments"] for u in a["utterances"])


def test_key_links_assignments_to_sources_and_repeats_come_last(study):
    key = json.loads((study / "sealed" / "key.json").read_text(encoding="utf-8"))
    r1 = next(p for p in key["packages"] if p["rater_id"] == "R1")
    assert [a["set"] for a in r1["assignments"]][-1] == "repeat"
    repeat = r1["assignments"][-1]
    original = next(a for a in r1["assignments"] if a["assignment_id"] == repeat["repeat_of"])
    assert original["snapshot_id"] == repeat["snapshot_id"]
    assert repeat["snapshot_id"]
    assert repeat["source"]["session_id"] == "session_20000101_000001_f1a001"
    r2 = next(p for p in key["packages"] if p["rater_id"] == "R2")
    assert len(r2["assignments"]) == 2


def test_a_study_version_is_written_once(study):
    with pytest.raises(BuildError, match="already exists"):
        build(FIXTURES / "manifest.json", study.parent)


def test_leaking_transcript_text_refuses_the_whole_build(tmp_path):
    copy_dir = tmp_path / "sessions" / SLEEP.name
    shutil.copytree(SLEEP, copy_dir)
    rows = [json.loads(line) for line in (copy_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["patient_text"] += " See session_20000101_000002_f1a002."
    (copy_dir / "transcript.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    manifest = {"study_version": "leak", "seed": 1, "raters": ["R1"], "sessions": [{"path": str(copy_dir)}]}
    (tmp_path / "m.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BuildError, match="reveal its sources"):
        build(tmp_path / "m.json", tmp_path / "out")
    assert not (tmp_path / "out" / "leak").exists(), "nothing may be written when a package fails the scan"


def returned(study: Path, rater: str = "R1") -> tuple[dict, dict]:
    """A complete, valid return for the sleep fixture session, as the page would write it."""
    pkg = package(study, rater)
    target = next(a for a in pkg["assignments"] if any("can't sleep" in u["text"] for u in a["utterances"]))
    patient = [u for u in target["utterances"] if u["speaker"] == "patient"]
    def span(u, phrase, occurrence=0):
        start = -1
        for _ in range(occurrence + 1):
            start = u["text"].index(phrase, start + 1)
        return {"utterance_id": u["id"], "start": start, "end": start + len(phrase), "quote": phrase}
    claims = [
        {"id": "c1", "utterance_id": patient[0]["id"], "symptom": "sleep_problems", "code": "present", "elicited": "no",
         "spans": [span(patient[0], "I can't sleep properly")], "context": [], "note": ""},
        {"id": "c2", "utterance_id": patient[1]["id"], "symptom": "sleep_problems", "code": "present", "elicited": "yes",
         "spans": [span(patient[1], "Most nights I lie awake")], "context": ["u02"], "note": "answers the sleep question"},
        {"id": "c3", "utterance_id": patient[2]["id"], "symptom": "low_energy", "code": "raised_and_denied", "elicited": "yes",
         "spans": [span(patient[2], "I'm not tired", 1)], "context": [], "note": ""},
        {"id": "c4", "utterance_id": patient[2]["id"], "symptom": "low_energy", "code": "present", "elicited": "unclear",
         "spans": [span(patient[2], "Energy's fine")], "context": [], "note": "contradiction kept"},
    ]
    form = {"expression": {k: "no_evidence" for k in KEYS} | {"sleep_problems": "present", "low_energy": "unclear"},
            "frequency": {k: "cannot_judge" for k in KEYS} | {"sleep_problems": "3"},
            "frequency_evidence": {"sleep_problems": ["c2"]}, "most_prominent": "sleep_problems",
            "presenting_complaint": "Garage project; poor sleep.", "realism": 4, "note": ""}
    turns = {u["id"]: {"reviewed": True, "reviewed_at": "2000-01-03T00:00:00Z"} for u in patient}
    record = {"turns": turns, "claims": claims, "form": form}
    document = {"format": RETURN_FORMAT, "schema_version": SCHEMA_VERSION, "app_version": pkg["app_version"],
                "package_id": pkg["package_id"], "study_version": pkg["study_version"],
                "protocol_version": pkg["protocol_version"], "rater_id": rater, "revision": 12,
                "saved_at": "2000-01-03T00:00:00Z",
                "assignments": {target["assignment_id"]: {"status": "submitted", **copy.deepcopy(record),
                                "submissions": [{"revision": 12, "submitted_at": "2000-01-03T00:00:00Z", **record}],
                                "reopenings": []}},
                "events": []}
    return document, target


def write(tmp_path: Path, name: str, document: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def test_a_valid_return_is_imported_and_reduced(study, tmp_path):
    document, target = returned(study)
    report = import_returns(study, [write(tmp_path, "r1.json", document)])
    assert [f["errors"] for f in report["files"]] == [[]]
    assert report["summary"]["claims"] == 4
    labels = [line.split(",") for line in (study / "returns" / "imported" / "turn_labels.csv").read_text().splitlines()[1:]]
    by_key = {(row[2], row[4]): (row[5], row[7]) for row in labels}
    patient = [u["id"] for u in target["utterances"] if u["speaker"] == "patient"]
    assert by_key[(patient[0], "sleep_problems")] == ("present", "False")
    assert by_key[(patient[2], "low_energy")] == ("unclear", "True"), "two codes on one turn reduce to unclear, flagged mixed"
    assert by_key[(patient[1], "appetite_changes")] == ("no_evidence", "False")
    assert report["raters_missing"] == ["R2"]
    assert list((study / "returns" / "raw" / "R1").glob("*.json")), "the returned file is kept unchanged"


@pytest.mark.parametrize("tamper, message", [
    (lambda d, a: d["assignments"][a]["submissions"][0]["claims"][0]["spans"][0].update(quote="I can sleep properly"), "not the quoted"),
    (lambda d, a: d["assignments"][a]["submissions"][0]["claims"][0].update(utterance_id="u00"), "not a patient turn"),
    (lambda d, a: d["assignments"][a]["submissions"][0]["claims"][0].update(symptom="insomnia"), "unknown symptom"),
    (lambda d, a: d["assignments"][a]["submissions"][0]["turns"].popitem(), "unreviewed"),
    (lambda d, a: d["assignments"][a]["submissions"][0]["form"]["frequency"].pop("appetite_changes"), "unanswered"),
    (lambda d, a: d.update(package_id="PNOTREAL"), "not part of this study"),
    (lambda d, a: d.update(rater_id="R2"), "rater_id"),
])
def test_tampered_returns_are_rejected(study, tmp_path, tamper, message):
    document, target = returned(study)
    tamper(document, target["assignment_id"])
    report = import_returns(study, [write(tmp_path, "bad.json", document)])
    errors = report["files"][0]["errors"]
    assert any(message in e for e in errors), errors
    assert report["summary"]["claims"] == 0, "a rejected file contributes nothing to the tables"


def test_newest_valid_file_wins(study, tmp_path):
    older, _ = returned(study)
    newer = copy.deepcopy(older)
    newer["revision"] = 20
    assignment_id = next(iter(newer["assignments"]))
    newer["assignments"][assignment_id]["submissions"][0]["form"]["realism"] = 2
    import_returns(study, [write(tmp_path, "old.json", older)])
    report = import_returns(study, [write(tmp_path, "new.json", newer)])
    used = [f for f in report["files"] if f["used"]]
    assert [f["revision"] for f in used] == [20]
    assert ",2," in (study / "returns" / "imported" / "session_globals.csv").read_text()


def selection(**overrides) -> argparse.Namespace:
    values = {"prefix": "fixture01", "sessions": [str(FIXTURES / "sessions" / "session_*")], "cases": None,
              "conditions": None, "runs": None, "shared": "1", "unique": None, "repeats": 0, "raters": "R1,R2",
              "seed": 7, "study_version": "sel", "description": "", "exclude_manifest": [], "allow_short": False}
    return argparse.Namespace(**(values | overrides))


def test_selector_draws_per_cell_and_records_the_frame():
    manifest = select(selection(repeats=1))
    assert sorted(manifest["selection"]["cells"]) == ["no_complaint/no_symptoms", "no_complaint/sleep_problems"]
    assert all(cell["frame"] == 1 and cell["p_shared"] == 1.0 for cell in manifest["selection"]["cells"].values())
    assert all(s["set"] == "shared" and s["raters"] == ["R1", "R2"] for s in manifest["sessions"])
    assert len(manifest["repeats"]) == 2, "one repeat per rater"


def test_selector_refuses_a_design_the_batch_cannot_meet():
    with pytest.raises(SelectionError, match="too small"):
        select(selection(shared="1", unique="1"))
    with pytest.raises(SelectionError, match="no eligible session"):
        select(selection(prefix="requested_19990101"))


def test_selector_output_builds(tmp_path):
    manifest = select(selection(conditions="sleep_problems"))
    (tmp_path / "m.json").write_text(json.dumps(manifest), encoding="utf-8")
    summary = build(tmp_path / "m.json", tmp_path / "out")
    assert [p["assignments"] for p in summary["packages"]] == [1, 1]
    excluded = select(selection(exclude_manifest=[tmp_path / "m.json"]))
    assert [s["cell"]["condition"] for s in excluded["sessions"]] == ["no_symptoms"], "excluded sessions stay out"


def test_case_names_combine_user_and_bundled_cases(tmp_path, monkeypatch):
    from dyadic_sim.core import paths
    from dyadic_sim.study_web import build_package

    cases = tmp_path / "config" / "priors" / "patient" / "cases"
    cases.mkdir(parents=True)
    (cases / "my_custom_case.yaml").write_text("presenting_complaint: x\n")
    monkeypatch.setattr(paths, "USER_CONFIG", str(tmp_path / "config"))

    names = build_package.case_names()
    assert "my_custom_case" in names and "no_complaint" in names
    # a run prefix in front of a bundled case is still found, so blinding still rejects it
    assert build_package.run_prefix("myrun_no_complaint_sleep_problems_run_1", names) == "myrun"
