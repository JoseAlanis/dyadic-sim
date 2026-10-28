"""Canonical input/output locations.

Layout. The primary split is raw vs derived; local vs external applies to raw:

  data/
  ├── sessions/                     RAW, generated on this machine
  │   └── session_*/
  ├── external/                     RAW, imported from elsewhere (read-only)
  │   ├── MANIFEST.yaml             origin + fingerprint per batch
  │   ├── _archives/                the zips the batches were extracted from
  │   └── <dir>/session_*/          one flat dir per batch
  ├── results/                      DERIVED, everything computed here
  │   └── <experiment>/
  └── _archive/                     bulky material kept but not in active use

Every raw session sits at exactly one level below its bucket, so one glob shape
reads any batch.

`external_sets()` is keyed by the manifest's `model` label, not by directory
name: those keys become the `model` column in the output CSVs, so they must stay
stable even if a directory is renamed.
"""
from __future__ import annotations

import glob
import hashlib
import os
import re

# Where commands read user configuration (config/) and write outputs (data/):
# the working directory. `dyadic-sim --root DIR` and DYADIC_SIM_ROOT change into
# DIR first.
ROOT = os.getcwd()

# Read-only defaults shipped with the package: the tracked config/ of the source
# tree, which the wheel carries as dyadic_sim/_defaults/config.
_PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WHEEL_DEFAULTS = os.path.join(_PACKAGE, "_defaults", "config")
BUNDLED_CONFIG = (_WHEEL_DEFAULTS if os.path.isdir(_WHEEL_DEFAULTS)
                  else os.path.join(os.path.dirname(_PACKAGE), "config"))
USER_CONFIG = os.path.join(ROOT, "config")


def config_path(*parts: str) -> str:
    """`<root>/config/<parts>` if it exists, else the bundled default."""
    user = os.path.join(USER_CONFIG, *parts)
    return user if os.path.exists(user) else os.path.join(BUNDLED_CONFIG, *parts)


def config_dirs(*parts: str) -> list[str]:
    """Existing config directories for listing, the user's first."""
    found = []
    for base in (USER_CONFIG, BUNDLED_CONFIG):
        path = os.path.realpath(os.path.join(base, *parts))
        if os.path.isdir(path) and path not in found:
            found.append(path)
    return found




def load_env() -> None:
    """Load `<root>/.env` (provider keys and settings); variables already set win."""
    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT, ".env"))


DATA = os.path.join(ROOT, "data")
SESSIONS = os.path.join(DATA, "sessions")           # local raw
EXTERNAL = os.path.join(DATA, "external")           # imported raw
ARCHIVE = os.path.join(DATA, "_archive")            # parked, not analysed
MANIFEST = os.path.join(EXTERNAL, "MANIFEST.yaml")

# local results
RESULTS = os.path.join(DATA, "results")

# The active corpus, and where a bare run of an analysis CLI reads and writes.
#
# These must be kept in AGREEMENT: sessions in, that corpus's results out. When
# they disagreed - reading one corpus while writing into another's results
# directory - a bare run with no arguments silently replaced a finished result
# set with something computed from entirely different sessions.
#
# The same hazard exists in the other direction: point these at a corpus that
# has been retired and a bare run recreates its directory, so a stale name comes
# back holding new numbers. Move both together or neither.
ACTIVE_PREFIX = "power01"
ACTIVE_SESSIONS = os.path.join(SESSIONS, "session_*")
ACTIVE_RESULTS = os.path.join(RESULTS, ACTIVE_PREFIX)


def _parse_manifest(path: str = MANIFEST) -> list[dict]:
    """Read the flat `- key: value` source list without a YAML dependency."""
    if not os.path.exists(path):
        return []
    sources: list[dict] = []
    current: dict | None = None
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        item = re.match(r"\s*-\s+(\w+):\s*(.*)", line)
        if item:
            if current:
                sources.append(current)
            current = {item.group(1): item.group(2).strip().strip('"')}
            continue
        field = re.match(r"\s{4,}(\w+):\s*(.*)", line)
        if field and current is not None:
            current[field.group(1)] = field.group(2).strip().strip('"')
    if current:
        sources.append(current)
    return sources


def external_sources(path: str = MANIFEST) -> list[dict]:
    """Manifest entries, each with an absolute `path` added."""
    sources = _parse_manifest(path)
    for source in sources:
        source["path"] = os.path.join(EXTERNAL, source["dir"])
    return sources


def _discover_external() -> list[dict]:
    """Fall back to directory discovery when the manifest is absent."""
    found = []
    for entry in sorted(os.listdir(EXTERNAL)) if os.path.isdir(EXTERNAL) else []:
        full = os.path.join(EXTERNAL, entry)
        if entry.startswith("_") or not os.path.isdir(full):
            continue
        found.append({"dir": entry, "model": entry, "path": full})
    return found


def external_sets(path: str = MANIFEST) -> dict[str, str]:
    """Model label -> glob over that batch's session directories."""
    sources = external_sources(path) or _discover_external()
    return {
        source.get("model", source["dir"]): os.path.join(source["path"], "session_*")
        for source in sources
    }


def fingerprint(directory: str) -> str:
    """Stable content hash of a session batch.

    Hashes sorted "session_id:sha256(transcript.jsonl)" lines, so it ignores
    mtimes and directory listing order and changes only if a transcript does.
    """
    digest = hashlib.sha256()
    for session in sorted(glob.glob(os.path.join(directory, "session_*"))):
        transcript = os.path.join(session, "transcript.jsonl")
        if os.path.exists(transcript):
            with open(transcript, "rb") as handle:
                inner = hashlib.sha256(handle.read()).hexdigest()
        else:
            inner = "MISSING"
        digest.update(f"{os.path.basename(session)}:{inner}\n".encode())
    return f"sha256:{digest.hexdigest()}"


def verify(path: str = MANIFEST) -> list[tuple[str, bool, str]]:
    """Check each batch still matches its recorded fingerprint."""
    report = []
    for source in external_sources(path):
        actual = fingerprint(source["path"])
        report.append((source["dir"], actual == source.get("fingerprint"), actual))
    return report


def main() -> int:
    """Check the imported corpora against their recorded fingerprints.

    `verify()` existed but nothing ever called it, so MANIFEST.yaml recorded a
    fingerprint per batch that was never checked: the integrity mechanism was
    write-only. Anything computed from these sessions is invalidated by a silent
    change to them, and without this check nothing would notice.

        dyadic-sim verify-external
    """
    report = verify()
    if not report:
        print(f"No external sources recorded in {MANIFEST}")
        return 0
    ok = True
    for directory, matches, actual in report:
        print(f"  {'OK  ' if matches else 'FAIL'}  {directory}")
        if not matches:
            ok = False
            print(f"        recorded fingerprint does not match; now {actual}")
    print("\nall batches match their manifest" if ok
          else "\nMISMATCH: results derived from these sessions are not reproducible")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
