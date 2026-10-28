"""The real page in headless Chrome: rate a session, save, and import the file.

Slow and needs Chrome plus Node 22 or newer, so it runs only when asked:

    STUDY_WEB_BROWSER=1 uv run pytest tests/study_web -q
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from study_web.build_package import build
from study_web.import_returns import run as import_returns

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    os.environ.get("STUDY_WEB_BROWSER") != "1" or not shutil.which("node")
    or not (shutil.which("google-chrome") or os.environ.get("CHROME")),
    reason="set STUDY_WEB_BROWSER=1, with Chrome and Node 22+ installed",
)


def test_browser_round_trip(tmp_path):
    summary = build(REPO / "study_web" / "fixtures" / "manifest.json", tmp_path)
    study = Path(summary["study_dir"])
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    result = subprocess.run(
        ["node", str(REPO / "tests" / "study_web" / "browser_e2e.mjs"), str(study / "packages" / "R1.html"), str(downloads)],
        capture_output=True, text=True, timeout=120, check=False,
    )
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome["ok"], outcome
    report = import_returns(study, [Path(outcome["file"])])
    assert [f["errors"] for f in report["files"]] == [[]], report["files"]
    assert report["summary"]["claims"] == 4
    quotes = (study / "returns" / "imported" / "claims.csv").read_text(encoding="utf-8")
    assert "I can't sleep properly" in quotes
