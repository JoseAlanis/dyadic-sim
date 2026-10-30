"""The installed package works without a checkout and ships only what it should.

The wheel carries the tracked config/ as bundled defaults and must never pick
up local or generated files, whether it is built directly or from the source
archive. Settings and credentials come from the selected root's `.env`, not
from wherever the package is installed. The full installed check (fresh
environment, imports, command help, bundled prompts) is
`tests/packaging/check_installed_wheel.sh`.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import dyadic_sim

REPO = Path(__file__).resolve().parents[1]
PACKAGE = Path(dyadic_sim.__file__).resolve().parent

MUST_SHIP = [
    "dyadic_sim/_defaults/config/models.yaml",
    "dyadic_sim/_defaults/config/priors/therapist/base.yaml",
    "dyadic_sim/_defaults/config/priors/patient/cases/no_complaint.yaml",
    "dyadic_sim/_defaults/config/priors/patient/symptom_facts.yaml",
    "dyadic_sim/_defaults/config/designs.example.yaml",
    "dyadic_sim/_defaults/config/analysis_spec.example.yaml",
    "dyadic_sim/analysis/exemplars/pool_a.jsonl",
    "dyadic_sim/study_web/page/index.html",
]
# Local or generated files planted in the copy before building.
PLANTED = [
    "config/designs.yaml",
    "config/analysis_spec.yaml",
    "config/priors/patient/cases/planted_no_complaint_sleep_problems_run_1.yaml",
    ".env",
    "data/sessions/session_planted/metadata.json",
    "manuscript/notes.md",
]


def _tracked_copy(dest: Path) -> None:
    files = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True,
                           check=True).stdout.split()
    for name in files:
        if (REPO / name).is_file():
            (dest / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, dest / name)
    for name in PLANTED:
        (dest / name).parent.mkdir(parents=True, exist_ok=True)
        (dest / name).write_text("planted: must not ship\n")


@pytest.mark.skipif(not shutil.which("uv") or not (REPO / ".git").exists(),
                    reason="needs uv and a git checkout")
@pytest.mark.parametrize("route", ["wheel", "sdist-then-wheel"])
def test_wheel_ships_bundled_defaults_and_no_local_files(tmp_path, route):
    source = tmp_path / "src"
    _tracked_copy(source)
    args = ["uv", "build", "-q", "--out-dir", str(tmp_path / "dist")]
    if route == "wheel":
        args.append("--wheel")
    subprocess.run(args, cwd=source, check=True, capture_output=True)
    (wheel,) = (tmp_path / "dist").glob("*.whl")
    names = zipfile.ZipFile(wheel).namelist()
    assert [n for n in MUST_SHIP if n not in names] == []
    planted = [n for n in names if "planted" in n or n.endswith(
        ("config/designs.yaml", "config/analysis_spec.yaml", ".env"))]
    assert planted == []
    assert [n for n in names if n.startswith(("tests/", "data/", "manuscript/"))] == []


def test_env_is_read_from_the_selected_root(tmp_path):
    (tmp_path / ".env").write_text("DYADIC_SIM_ENV_PROBE=from-root\n")
    env = {k: v for k, v in os.environ.items() if k != "DYADIC_SIM_ENV_PROBE"}
    probe = ("from dyadic_sim.core import paths; paths.load_env(); "
             "import os; print(os.environ.get('DYADIC_SIM_ENV_PROBE'))")
    out = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, env=env,
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == "from-root"


def test_only_core_paths_loads_env_files():
    callers = []
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "load_dotenv"):
                callers.append((path.relative_to(PACKAGE).as_posix(), len(node.args)))
    assert callers == [("core/paths.py", 1)]
