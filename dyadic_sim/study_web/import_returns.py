"""Validate returned rating files against their sealed packages, then tabulate.

    uv run dyadic-sim import-returns \
        --study-dir data/results/clinician_study/<study_version> <file.json> [...]

Nothing in a returned file is trusted. Each is checked against sealed/key.json
and the frozen snapshots:

- package, rater, study and protocol versions match what was sent;
- every assignment, utterance and claim reference exists;
- every evidence span lies inside a patient utterance, and `text[start:end]`
  (code points) equals its quote;
- every submitted session has all patient turns reviewed and every form field
  answered, "cannot judge" included.

Returned files are copied unchanged into returns/raw/. The tables in
returns/imported/ are rebuilt on every run from the newest valid file per rater,
and carry opaque ids only; join sealed/key.json for sources and conditions.

Turn-by-symptom labels use reduction rule `turn-v0-draft`, to be fixed in
calibration: unreviewed turn -> missing; reviewed without a claim for the
symptom -> no_evidence; claims with one code -> that code; claims with several
different codes -> unclear, with mixed=True. The original claims are kept.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from dyadic_sim.study_web import RETURN_FORMAT, SCHEMA_VERSION, codebook
from dyadic_sim.study_web.snapshot import sha256_bytes

REDUCTION_RULE = "turn-v0-draft"
STATUSES = {"not_started", "in_progress", "submitted"}
SYMPTOM_KEYS = [key for key, _, _ in codebook.SYMPTOMS]


class Problems:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, where: str, message: str) -> None:
        self.errors.append(f"{where}: {message}")

    def warn(self, where: str, message: str) -> None:
        self.warnings.append(f"{where}: {message}")


def load_study(study_dir: Path) -> dict:
    key = json.loads((study_dir / "sealed" / "key.json").read_text(encoding="utf-8"))
    snapshots = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in (study_dir / "sealed" / "snapshots").glob("*.json")
    }
    packages = {}
    for package in key["packages"]:
        assignments = {}
        for item in package["assignments"]:
            utterances = snapshots[item["snapshot_id"]]["utterances"]
            assignments[item["assignment_id"]] = {
                **item,
                "utterances": {u["id"]: u for u in utterances},
                "position": {u["id"]: i for i, u in enumerate(utterances)},
                "patient": [u["id"] for u in utterances if u["speaker"] == "patient"],
            }
        packages[package["package_id"]] = {**package, "assignments": assignments}
    return {"key": key, "packages": packages}


def check_claim(claim, assignment: dict, problems: Problems, where: str) -> None:
    if not isinstance(claim, dict):
        problems.error(where, "claim is not an object")
        return
    for name, kind in (("id", str), ("utterance_id", str), ("symptom", str), ("code", str),
                       ("elicited", str), ("spans", list), ("context", list), ("note", str)):
        if not isinstance(claim.get(name), kind):
            problems.error(where, f"claim field {name!r} is missing or has the wrong type")
            return
    where = f"{where} claim {claim['id']}"
    utterance = assignment["utterances"].get(claim["utterance_id"])
    if utterance is None or utterance["speaker"] != "patient":
        problems.error(where, f"utterance {claim['utterance_id']!r} is not a patient turn of this session")
        return
    if claim["symptom"] not in SYMPTOM_KEYS:
        problems.error(where, f"unknown symptom {claim['symptom']!r}")
    if claim["code"] not in codebook.keys(codebook.CLAIM_CODES):
        problems.error(where, f"unknown code {claim['code']!r}")
    if claim["elicited"] not in codebook.keys(codebook.ELICITED):
        problems.error(where, f"unknown elicited value {claim['elicited']!r}")
    if not claim["spans"]:
        problems.error(where, "no evidence span")
    text = utterance["text"]
    for span in claim["spans"]:
        if not isinstance(span, dict) or span.get("utterance_id") != claim["utterance_id"]:
            problems.error(where, "a span points outside the claim's own turn")
            continue
        start, end, quote = span.get("start"), span.get("end"), span.get("quote")
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text)):
            problems.error(where, f"span [{start}, {end}) is outside the turn (length {len(text)})")
        elif text[start:end] != quote:
            problems.error(where, f"span [{start}, {end}) reads {text[start:end]!r}, not the quoted {quote!r}")
    for uid in claim["context"]:
        if uid not in assignment["utterances"] or uid == claim["utterance_id"]:
            problems.error(where, f"context link {uid!r} is not another utterance of this session")


def check_form(form, claims: dict, complete: bool, problems: Problems, where: str) -> None:
    if not isinstance(form, dict):
        problems.error(where, "form is missing")
        return
    options = {
        "expression": codebook.keys(codebook.EXPRESSION),
        "frequency": codebook.keys(codebook.FREQUENCY),
    }
    for name, allowed in options.items():
        answers = form.get(name) or {}
        for symptom in SYMPTOM_KEYS:
            value = answers.get(symptom)
            if value is None:
                if complete:
                    problems.error(where, f"{name} for {symptom} is unanswered")
            elif value not in allowed:
                problems.error(where, f"{name} for {symptom} has invalid value {value!r}")
        for extra in set(answers) - set(SYMPTOM_KEYS):
            problems.error(where, f"{name} has an unknown symptom {extra!r}")
    for symptom, linked in (form.get("frequency_evidence") or {}).items():
        for claim_id in linked or []:
            if claims.get(claim_id, {}).get("symptom") != symptom:
                problems.error(where, f"frequency evidence for {symptom} links {claim_id!r}, which is not a claim about it")
    prominent = form.get("most_prominent")
    if prominent is None:
        if complete:
            problems.error(where, "most prominent symptom is unanswered")
    elif prominent not in set(SYMPTOM_KEYS) | codebook.keys(codebook.PROMINENCE_EXTRA):
        problems.error(where, f"most prominent has invalid value {prominent!r}")
    realism = form.get("realism")
    if realism is None:
        if complete:
            problems.error(where, "realism is unanswered")
    elif realism not in codebook.keys(codebook.REALISM):
        problems.error(where, f"realism has invalid value {realism!r}")
    complaint = form.get("presenting_complaint")
    if complete and not (isinstance(complaint, str) and complaint.strip()):
        problems.error(where, "presenting complaint is empty")


def check_record(record, assignment: dict, complete: bool, problems: Problems, where: str) -> None:
    """Turn reviews, claims and form of one working state or one submission."""
    turns = record.get("turns") or {}
    for uid in turns:
        if uid not in assignment["patient"]:
            problems.error(where, f"review mark on {uid!r}, which is not a patient turn")
    claims = record.get("claims") or []
    seen: dict[str, dict] = {}
    for claim in claims:
        check_claim(claim, assignment, problems, where)
        if isinstance(claim, dict) and isinstance(claim.get("id"), str):
            if claim["id"] in seen:
                problems.error(where, f"claim id {claim['id']!r} is used twice")
            seen[claim["id"]] = claim
    check_form(record.get("form"), seen, complete, problems, where)
    if complete:
        missing = [uid for uid in assignment["patient"] if not (turns.get(uid) or {}).get("reviewed")]
        if missing:
            problems.error(where, f"submitted with {len(missing)} patient turn(s) unreviewed")


def check_file(document, study: dict, problems: Problems) -> dict | None:
    """The package this file belongs to, or None if it cannot be matched."""
    if not isinstance(document, dict) or document.get("format") != RETURN_FORMAT:
        problems.error("file", "not a rating return file")
        return None
    if document.get("schema_version") != SCHEMA_VERSION:
        problems.error("file", f"schema version {document.get('schema_version')!r}, expected {SCHEMA_VERSION}")
        return None
    package = study["packages"].get(document.get("package_id"))
    if package is None:
        problems.error("file", f"package {document.get('package_id')!r} is not part of this study")
        return None
    key = study["key"]
    for name, expected in (("rater_id", package["rater_id"]), ("study_version", key["study_version"]),
                           ("protocol_version", key["protocol_version"])):
        if document.get(name) != expected:
            problems.error("file", f"{name} is {document.get(name)!r}, expected {expected!r}")
    if not isinstance(document.get("revision"), int):
        problems.error("file", "revision is missing")
    for assignment_id, state in (document.get("assignments") or {}).items():
        where = f"assignment {assignment_id}"
        assignment = package["assignments"].get(assignment_id)
        if assignment is None:
            problems.error(where, "not an assignment of this package")
            continue
        if state.get("status") not in STATUSES:
            problems.error(where, f"unknown status {state.get('status')!r}")
        check_record(state, assignment, False, problems, where)
        submissions = state.get("submissions") or []
        for number, submission in enumerate(submissions, 1):
            check_record(submission, assignment, True, problems, f"{where} submission {number}")
        if state.get("status") == "submitted" and not submissions:
            problems.error(where, "marked submitted but holds no submission")
        for reopening in state.get("reopenings") or []:
            if not (isinstance(reopening, dict) and str(reopening.get("reason", "")).strip()):
                problems.error(where, "a reopening has no reason")
    return package


def turn_labels(submission: dict, assignment: dict) -> list[dict]:
    """Reduction rule turn-v0-draft; see the module docstring."""
    rows = []
    reviewed = {uid for uid, mark in (submission.get("turns") or {}).items() if (mark or {}).get("reviewed")}
    for uid in assignment["patient"]:
        for symptom in SYMPTOM_KEYS:
            codes = [c["code"] for c in submission["claims"] if c["utterance_id"] == uid and c["symptom"] == symptom]
            if uid not in reviewed:
                label, mixed = "missing", False
            elif not codes:
                label, mixed = "no_evidence", False
            elif len(set(codes)) == 1:
                label, mixed = codes[0], False
            else:
                label, mixed = "unclear", True
            rows.append({"utterance_id": uid, "turn": assignment["utterances"][uid]["turn"],
                         "symptom": symptom, "label": label, "n_claims": len(codes), "mixed": mixed})
    return rows


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def tabulate(chosen: list[tuple[dict, dict, dict]], out: Path) -> dict:
    """Canonical JSONL plus derived tables, from each rater's newest valid file."""
    out.mkdir(parents=True, exist_ok=True)
    claims, labels, forms, globals_, lines = [], [], [], [], []
    counts = {"submitted": 0, "in_progress": 0, "not_started": 0}
    for document, package, meta in chosen:
        rater = package["rater_id"]
        for assignment_id, assignment in sorted(package["assignments"].items(), key=lambda kv: kv[1]["order"]):
            state = (document.get("assignments") or {}).get(assignment_id) or {"status": "not_started"}
            status = state.get("status", "not_started")
            counts[status] = counts.get(status, 0) + 1
            submissions = state.get("submissions") or []
            latest = submissions[-1] if status == "submitted" and submissions else None
            lines.append({
                "rater_id": rater, "package_id": package["package_id"], "assignment_id": assignment_id,
                "order": assignment["order"], "status": status, "file_sha256": meta["sha256"],
                "file_revision": document.get("revision"), "reduction_rule": REDUCTION_RULE,
                "n_submissions": len(submissions), "reopenings": state.get("reopenings") or [],
                "submission": latest,
            })
            if latest is None:
                continue
            base = {"rater_id": rater, "assignment_id": assignment_id}
            for claim in latest["claims"]:
                claims.append({
                    **base, "claim_id": claim["id"], "utterance_id": claim["utterance_id"],
                    "turn": assignment["utterances"][claim["utterance_id"]]["turn"],
                    "symptom": claim["symptom"], "code": claim["code"], "elicited": claim["elicited"],
                    "n_spans": len(claim["spans"]), "quotes": " | ".join(s["quote"] for s in claim["spans"]),
                    "spans": json.dumps([[s["start"], s["end"]] for s in claim["spans"]]),
                    "context": " ".join(claim["context"]), "note": claim["note"],
                })
            labels += [{**base, **row} for row in turn_labels(latest, assignment)]
            form = latest["form"]
            for symptom in SYMPTOM_KEYS:
                forms.append({**base, "symptom": symptom,
                              "expression": form["expression"].get(symptom),
                              "frequency": form["frequency"].get(symptom),
                              "frequency_evidence": " ".join((form.get("frequency_evidence") or {}).get(symptom) or [])})
            globals_.append({**base, "most_prominent": form.get("most_prominent"),
                             "presenting_complaint": form.get("presenting_complaint"),
                             "realism": form.get("realism"), "note": form.get("note", ""),
                             "submitted_at": latest.get("submitted_at"), "submission_number": len(submissions),
                             "n_reopenings": len(state.get("reopenings") or [])})
    with (out / "ratings.jsonl").open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    write_csv(out / "claims.csv", claims, ["rater_id", "assignment_id", "claim_id", "utterance_id", "turn",
              "symptom", "code", "elicited", "n_spans", "quotes", "spans", "context", "note"])
    write_csv(out / "turn_labels.csv", labels, ["rater_id", "assignment_id", "utterance_id", "turn",
              "symptom", "label", "n_claims", "mixed"])
    write_csv(out / "session_symptoms.csv", forms, ["rater_id", "assignment_id", "symptom",
              "expression", "frequency", "frequency_evidence"])
    write_csv(out / "session_globals.csv", globals_, ["rater_id", "assignment_id", "most_prominent",
              "presenting_complaint", "realism", "note", "submitted_at", "submission_number", "n_reopenings"])
    return {"assignments": counts, "claims": len(claims), "turn_label_rows": len(labels)}


def run(study_dir: Path, files: list[Path]) -> dict:
    study = load_study(study_dir)
    raw_dir = study_dir / "returns" / "raw"
    report_files, candidates = [], {}
    # Files already imported are reconsidered too, so one bad new file cannot hide an older good one.
    previous = sorted(raw_dir.glob("*/*.json")) if raw_dir.is_dir() else []
    seen_hashes = set()
    for path in [*files, *previous]:
        data = path.read_bytes()
        sha = sha256_bytes(data)
        if sha in seen_hashes:
            continue
        seen_hashes.add(sha)
        problems = Problems()
        try:
            document = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            document, package = None, None
            problems.error("file", f"not valid JSON: {exc}")
        else:
            package = check_file(document, study, problems)
        rater = package["rater_id"] if package else "unknown"
        stored = raw_dir / rater / f"rev{(document or {}).get('revision', 'x')}_{sha[:12]}.json"
        if not stored.exists():
            stored.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, stored)
        entry = {"file": str(path), "stored_as": str(stored.relative_to(study_dir)), "sha256": sha,
                 "rater_id": rater, "revision": (document or {}).get("revision"),
                 "saved_at": (document or {}).get("saved_at"),
                 "errors": problems.errors, "warnings": problems.warnings, "used": False}
        report_files.append(entry)
        if package and not problems.errors:
            best = candidates.get(rater)
            if best is None or document["revision"] > best[0]["revision"]:
                candidates[rater] = (document, package, entry)
            elif document["revision"] == best[0]["revision"] and sha != best[2]["sha256"]:
                best[2]["warnings"].append(f"another file has the same revision {document['revision']} but different content")
    for _, _, entry in candidates.values():
        entry["used"] = True
    summary = tabulate([(d, p, {"sha256": e["sha256"]}) for d, p, e in candidates.values()],
                       study_dir / "returns" / "imported")
    report = {"study_version": study["key"]["study_version"], "imported_at": datetime.now(UTC).isoformat(timespec="seconds"),
              "reduction_rule": REDUCTION_RULE, "files": report_files, "summary": summary,
              "raters_missing": sorted({p["rater_id"] for p in study["packages"].values()} - set(candidates))}
    (study_dir / "returns" / "imported" / "import_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()
    report = run(args.study_dir, args.files)
    bad = [f for f in report["files"] if f["errors"]]
    for entry in report["files"]:
        state = "used" if entry["used"] else ("REJECTED" if entry["errors"] else "older, kept")
        print(f"  {state:12s} {entry['rater_id']:8s} rev {entry['revision']!s:>4}  {entry['file']}")
        for message in entry["errors"][:10]:
            print(f"      error: {message}")
        for message in entry["warnings"][:5]:
            print(f"      warning: {message}")
    print(json.dumps(report["summary"], indent=1))
    if report["raters_missing"]:
        print(f"no valid file yet from: {', '.join(report['raters_missing'])}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
