"""Build blinded, self-contained rating packages: one HTML file per rater.

    uv run python -m study_web.build_package \
        --manifest data/results/clinician_study/manifests/<study_version>.json

The manifest is an explicit allowlist of session directories, never a glob; its
format is in study_web/README.md. Output, under --out-root (default
data/results/clinician_study):

  <study_version>/packages/<rater>.html        the only thing a rater receives
  <study_version>/packages/MANIFEST.json       package ids and file hashes
  <study_version>/sealed/key.json              unblinding key: never send it
  <study_version>/sealed/snapshots/<id>.json   exact displayed text, for import

Blinding is enforced twice. The page payload is assembled from an allowlist of
fields, and each finished HTML file is scanned for anything that identifies a
source session: session ids, case names, run prefixes, run numbers, file paths
and generation timestamps. If any package fails, nothing is written.

A study version is written once. Returned files are bound to the package ids
inside it, so rebuilding into an existing directory is refused.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from study_web import APP_VERSION, PACKAGE_FORMAT, SCHEMA_VERSION, codebook
from study_web.snapshot import (
    PREPROCESSING,
    dialogue,
    digest,
    patient_ids,
    read_session,
    sha256_bytes,
)

REPO = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).resolve().parent / "page"
CASES_DIR = REPO / "config" / "priors" / "patient" / "cases"
DEFAULT_OUT = REPO / "data" / "results" / "clinician_study"
ID_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: The only fields the page payload may carry, level by level.
PAYLOAD_FIELDS = {
    "format", "schema_version", "app_version", "package_id", "study_version",
    "protocol_version", "rater_id", "built_at", "codebook", "assignments",
}
ASSIGNMENT_FIELDS = {"assignment_id", "order", "utterances"}
UTTERANCE_FIELDS = {"id", "speaker", "turn", "text"}

#: Patterns that identify a source session wherever they appear in a package.
LEAK_PATTERNS = [
    re.compile(r"session_\d{8}_\d{6}_[0-9a-f]{6}"),
    re.compile(r"_run_\d+"),
    re.compile(r"data/sessions|data/external|transcript\.jsonl|metadata\.json|_transcript\.pdf"),
    re.compile(r"patient_symptom|case_name|symptom_levels|initial_patient_prompt"),
]


class BuildError(RuntimeError):
    """The manifest or a package is not safe to write."""


@dataclass
class Spec:
    study_version: str
    protocol_version: str
    seed: int
    raters: list[str]
    sessions: list[dict]
    repeats: list[dict] = field(default_factory=list)
    description: str = ""


def case_names() -> list[str]:
    """Hand-written case names; generated per-condition files carry `_run_`."""
    return sorted(
        path.stem for path in CASES_DIR.glob("*.yaml")
        if not path.stem.startswith("_") and "_run_" not in path.stem
    )


def run_prefix(case_name: str, cases: list[str]) -> str | None:
    """The run prefix before the case marker, as analysis.validation.sessions reads it."""
    for case in sorted(cases, key=len, reverse=True):
        index = case_name.find(case)
        if index > 0:
            return case_name[:index].rstrip("_") or None
    return None


def load_spec(manifest: dict) -> Spec:
    try:
        spec = Spec(
            study_version=manifest["study_version"],
            protocol_version=manifest.get("protocol_version", codebook.PROTOCOL_VERSION),
            seed=manifest["seed"],
            raters=list(manifest["raters"]),
            sessions=list(manifest["sessions"]),
            repeats=list(manifest.get("repeats", [])),
            description=manifest.get("description", ""),
        )
    except (KeyError, TypeError) as exc:
        raise BuildError(f"manifest is missing or mistypes a field: {exc}") from exc
    if not isinstance(spec.study_version, str) or not SAFE_NAME.match(spec.study_version):
        raise BuildError("study_version must be a plain name: letters, digits, '.', '_' or '-'")
    if spec.protocol_version != codebook.PROTOCOL_VERSION:
        raise BuildError(
            f"manifest asks for protocol {spec.protocol_version!r}, but the codebook in "
            f"study_web/codebook.py is {codebook.PROTOCOL_VERSION!r}"
        )
    if not isinstance(spec.seed, int):
        raise BuildError("seed must be an integer")
    if not spec.raters or len(set(spec.raters)) != len(spec.raters):
        raise BuildError("raters must be a non-empty list of unique ids")
    for rater in spec.raters:
        if not isinstance(rater, str) or not SAFE_NAME.match(rater):
            raise BuildError(f"rater id {rater!r} must be a plain pseudonym such as R1")
    if not spec.sessions:
        raise BuildError("the manifest lists no sessions")
    paths = [entry.get("path") for entry in spec.sessions]
    if len(set(paths)) != len(paths):
        raise BuildError("a session path is listed twice; use `repeats` for deliberate re-rating")
    for entry in spec.sessions:
        unknown = set(entry.get("raters", spec.raters)) - set(spec.raters)
        if unknown:
            raise BuildError(f"{entry.get('path')}: unknown raters {sorted(unknown)}")
    for entry in spec.repeats:
        if entry.get("rater") not in spec.raters:
            raise BuildError(f"repeat {entry.get('path')}: unknown rater {entry.get('rater')!r}")
        original = next((s for s in spec.sessions if s.get("path") == entry.get("path")), None)
        if original is None or entry["rater"] not in original.get("raters", spec.raters):
            raise BuildError(f"repeat {entry.get('path')}: {entry['rater']} has no original assignment of it")
    return spec


def resolve(path: str) -> Path:
    candidate = Path(path)
    candidate = candidate if candidate.is_absolute() else REPO / candidate
    if not (candidate / "metadata.json").is_file() or not (candidate / "transcript.jsonl").is_file():
        raise BuildError(f"{path}: not a session directory (needs metadata.json and transcript.jsonl)")
    return candidate


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path.resolve())


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


class Ids:
    """Opaque ids from a seeded generator: reproducible, and meaningless without the key."""

    def __init__(self, seed: str) -> None:
        self.rng = random.Random(seed)
        self.used: set[str] = set()

    def new(self, prefix: str, length: int) -> str:
        while True:
            value = prefix + "".join(self.rng.choices(ID_ALPHABET, k=length))
            if value not in self.used:
                self.used.add(value)
                return value


def check_payload(payload: dict) -> None:
    """Refuse any field outside the allowlist, at every level."""
    extra = set(payload) - PAYLOAD_FIELDS
    if extra:
        raise BuildError(f"payload carries unexpected fields {sorted(extra)}")
    for assignment in payload["assignments"]:
        if set(assignment) - ASSIGNMENT_FIELDS:
            raise BuildError(f"assignment carries unexpected fields {sorted(set(assignment) - ASSIGNMENT_FIELDS)}")
        for utterance in assignment["utterances"]:
            if set(utterance) != UTTERANCE_FIELDS:
                raise BuildError(f"utterance fields {sorted(utterance)} differ from {sorted(UTTERANCE_FIELDS)}")


def scan(document: str, literals: set[str]) -> list[str]:
    """Everything in a finished package that could identify a source session."""
    hits = [f"pattern {p.pattern!r}: {m.group(0)!r}" for p in LEAK_PATTERNS for m in [p.search(document)] if m]
    hits += [f"literal {value!r}" for value in sorted(literals) if value and value in document]
    return hits


def page_parts() -> dict[str, str]:
    parts = {
        "THEME": (PAGE / "theme.css").read_text(encoding="utf-8"),
        "STYLE": (PAGE / "style.css").read_text(encoding="utf-8"),
        "APP": (PAGE / "app.js").read_text(encoding="utf-8"),
    }
    for name, text in parts.items():
        if "</script" in text.lower() or "</style" in text.lower():
            raise BuildError(f"page/{name.lower()} contains a closing tag that would break the page")
    return parts


def render(payload: dict, parts: dict[str, str]) -> str:
    """Fill the template in one pass, so inserted text is never itself rescanned for markers."""
    template = (PAGE / "index.html").read_text(encoding="utf-8")
    values = {
        **parts,
        # "<" is escaped so no transcript text can close the script element.
        "DATA": json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c"),
        "RATER": html.escape(payload["rater_id"]),
    }
    for name in values:
        if template.count(f"__{name}__") != 1:
            raise BuildError(f"page/index.html must contain __{name}__ exactly once")
    return re.sub(r"__(THEME|STYLE|APP|DATA|RATER)__", lambda m: values[m.group(1)], template)


def build(manifest_path: Path, out_root: Path = DEFAULT_OUT, now: str | None = None) -> dict:
    manifest_bytes = manifest_path.read_bytes()
    spec = load_spec(json.loads(manifest_bytes))
    study_dir = out_root / spec.study_version
    if study_dir.exists():
        raise BuildError(f"{study_dir} already exists. A study version is written once; choose a new study_version.")

    cases = case_names()
    ids = Ids(f"{spec.seed}|{spec.study_version}")
    literals = set(cases)
    snapshots: dict[str, dict] = {}
    for path in [entry["path"] for entry in spec.sessions]:
        directory = resolve(path)
        meta, rows, hashes = read_session(directory)
        if not meta.get("turn_count"):
            raise BuildError(f"{path}: turn_count is 0, so the session is incomplete")
        shown, sources, notes = dialogue(meta, rows)
        if not patient_ids(shown):
            raise BuildError(f"{path}: no patient turns")
        case_name = meta.get("case_name", "")
        source = {
            "session_id": meta.get("session_id"),
            "dir": relative(directory),
            "case_name": case_name,
            "run_prefix": run_prefix(case_name, cases),
            "generation_version": meta.get("generation_version"),
            "created_at": meta.get("created_at"),
            **hashes,
        }
        literals |= {
            str(value) for value in (
                source["session_id"], directory.name, case_name, source["run_prefix"],
                meta.get("created_at"), meta.get("updated_at"),
                Path(meta.get("pdf_transcript_path") or "").name,
            ) if value
        }
        snapshots[path] = {
            "snapshot_id": ids.new("S", 8), "digest": digest(shown), "preprocessing": PREPROCESSING,
            "source": source, "notes": notes, "shown": shown,
            "utterances": [{**u, "source": s} for u, s in zip(shown, sources, strict=True)],
        }

    built_at = now or datetime.now(UTC).isoformat(timespec="seconds")
    parts = page_parts()
    packages, key_packages = [], []
    for rater in spec.raters:
        main = [entry for entry in spec.sessions if rater in entry.get("raters", spec.raters)]
        random.Random(f"{spec.seed}|{spec.study_version}|{rater}").shuffle(main)
        repeats = [entry for entry in spec.repeats if entry["rater"] == rater]
        random.Random(f"{spec.seed}|{spec.study_version}|{rater}|repeats").shuffle(repeats)
        package_id = ids.new("P", 6)
        payload_assignments, key_assignments, first_of = [], [], {}
        # Repeats come last: a drift check re-rates at the end, under a fresh id.
        for order, (entry, is_repeat) in enumerate([(e, False) for e in main] + [(e, True) for e in repeats], 1):
            snapshot = snapshots[entry["path"]]
            assignment_id = ids.new("A", 6)
            first_of.setdefault(entry["path"], assignment_id)
            payload_assignments.append({"assignment_id": assignment_id, "order": order, "utterances": snapshot["shown"]})
            key_assignments.append({
                "assignment_id": assignment_id, "order": order, "snapshot_id": snapshot["snapshot_id"],
                "set": entry.get("set", "repeat" if is_repeat else "main"),
                "repeat_of": first_of[entry["path"]] if is_repeat else None,
                "source": snapshot["source"],
            })
        payload = {
            "format": PACKAGE_FORMAT, "schema_version": SCHEMA_VERSION, "app_version": APP_VERSION,
            "package_id": package_id, "study_version": spec.study_version,
            "protocol_version": spec.protocol_version, "rater_id": rater, "built_at": built_at,
            "codebook": codebook.as_payload(), "assignments": payload_assignments,
        }
        check_payload(payload)
        document = render(payload, parts)
        hits = scan(document, literals)
        if hits:
            raise BuildError(f"package for {rater} would reveal its sources: " + "; ".join(hits[:5]))
        packages.append((rater, package_id, document, len(payload_assignments)))
        key_packages.append({"rater_id": rater, "package_id": package_id, "assignments": key_assignments})

    # Every package passed; only now is anything written.
    (study_dir / "packages").mkdir(parents=True)
    (study_dir / "sealed" / "snapshots").mkdir(parents=True)
    for snapshot in snapshots.values():
        record = {k: snapshot[k] for k in ("snapshot_id", "digest", "preprocessing", "source", "notes", "utterances")}
        (study_dir / "sealed" / "snapshots" / f"{snapshot['snapshot_id']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    listing = []
    for (rater, package_id, document, count), key_package in zip(packages, key_packages, strict=True):
        file = study_dir / "packages" / f"{rater}.html"
        file.write_text(document, encoding="utf-8")
        key_package["file"] = f"packages/{file.name}"
        key_package["file_sha256"] = sha256_bytes(file.read_bytes())
        listing.append({"rater_id": rater, "package_id": package_id, "file": file.name,
                        "sha256": key_package["file_sha256"], "assignments": count})
    common = {"study_version": spec.study_version, "protocol_version": spec.protocol_version,
              "app_version": APP_VERSION, "schema_version": SCHEMA_VERSION, "built_at": built_at}
    (study_dir / "packages" / "MANIFEST.json").write_text(
        json.dumps({**common, "packages": listing}, indent=1), encoding="utf-8")
    key = {
        **common, "description": spec.description, "seed": spec.seed, "code": git_state(),
        "manifest_sha256": sha256_bytes(manifest_bytes), "manifest": json.loads(manifest_bytes),
        "packages": key_packages,
    }
    (study_dir / "sealed" / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"study_dir": str(study_dir), "packages": listing, "snapshots": len(snapshots)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    try:
        summary = build(args.manifest, args.out_root)
    except BuildError as exc:
        print(f"build refused: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
