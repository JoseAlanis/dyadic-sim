#!/usr/bin/env bash
# Installed-wheel check: build the wheel, install it with the locked
# dependencies into a fresh environment, and use it from an unrelated directory
# without PYTHONPATH. Run from the repository root:
#   tests/packaging/check_installed_wheel.sh
# Wheel contents for both build routes are also tested in tests/test_packaging.py.
set -euo pipefail
REPO=$(pwd)
W=$(mktemp -d -t dyadic-wheel-XXXX); trap 'rm -rf "$W"' EXIT
unset PYTHONPATH
fail() { echo "FAIL: $*"; exit 1; }

echo "== build (sdist, then wheel from the sdist)"
uv build -q --out-dir "$W/dist" >/dev/null
WHEEL=$(ls "$W"/dist/*.whl)
python3 -I - "$WHEEL" <<'PY' || fail "wheel contents"
import sys, zipfile
names = zipfile.ZipFile(sys.argv[1]).namelist()
need = ["dyadic_sim/_defaults/config/models.yaml",
        "dyadic_sim/_defaults/config/priors/therapist/base.yaml",
        "dyadic_sim/_defaults/config/priors/patient/cases/no_complaint.yaml",
        "dyadic_sim/_defaults/config/priors/patient/symptom_facts.yaml",
        "dyadic_sim/_defaults/config/designs.example.yaml",
        "dyadic_sim/_defaults/config/analysis_spec.example.yaml",
        "dyadic_sim/analysis/exemplars/pool_a.jsonl",
        "dyadic_sim/study_web/page/index.html"]
missing = [n for n in need if n not in names]
bad = [n for n in names if n.endswith(("config/designs.yaml", "config/analysis_spec.yaml"))
       or "_run_" in n.rsplit("/", 1)[-1] or n.startswith(("tests/", "manuscript/", "data/"))
       or "/.env" in n or n.endswith(".env")]
print(f"   {len(names)} files; missing={missing}; must-not-ship={bad}")
sys.exit(1 if missing or bad else 0)
PY

echo "== fresh environment with the locked dependencies"
# A copy of pyproject.toml + uv.lock only (no source), synced into a new venv:
# the locked dependencies (dev group included, for pytest), not the project.
mkdir "$W/lock" && cp pyproject.toml uv.lock README.md "$W/lock/"
UV_PROJECT_ENVIRONMENT="$W/venv" uv sync -q --locked --no-install-project --project "$W/lock"
uv pip install -q --python "$W/venv" --no-deps "$WHEEL"

mkdir "$W/elsewhere"; cd "$W/elsewhere"
PY="$W/venv/bin/python"

echo "== every module imports from the installed package"
"$PY" - <<'PY' || fail "imports"
import importlib, pkgutil, dyadic_sim
assert "site-packages" in dyadic_sim.__file__, dyadic_sim.__file__
mods = [m.name for m in pkgutil.walk_packages(dyadic_sim.__path__, "dyadic_sim.")]
for name in mods:
    importlib.import_module(name)
print(f"   {len(mods)} modules from {dyadic_sim.__file__.rsplit('/dyadic_sim/', 1)[0]}")
PY

echo "== CLI help"
"$W/venv/bin/dyadic-sim" --help >/dev/null || fail "dyadic-sim --help"
for c in $("$PY" -c "from dyadic_sim.cli import COMMANDS; print(' '.join(COMMANDS))"); do
  "$W/venv/bin/dyadic-sim" "$c" --help >/dev/null 2>&1 || fail "dyadic-sim $c --help"
done
echo "   dyadic-sim and every command answer --help"

echo "== bundled resources"
"$PY" - <<'PY' || fail "resources"
from dyadic_sim.core import paths
from dyadic_sim.generation.priors.loader import list_cases, list_orientations
from dyadic_sim.analysis import exemplars
from dyadic_sim.study_web import build_package, codebook
assert paths.BUNDLED_CONFIG.endswith("_defaults/config"), paths.BUNDLED_CONFIG
assert not __import__("os").path.isdir(paths.USER_CONFIG), "no user config expected here"
cases, variants = list_cases(), list_orientations()
pools = exemplars.available()
assert cases and variants and pools and build_package.PAGE.joinpath("index.html").is_file()
assert codebook.PROTOCOL_VERSION
print(f"   cases={len(cases)} variants={len(variants)} pools={len(pools)} page+codebook ok")
PY

echo "== --root and .env from the selected root"
mkdir "$W/root" && printf 'DYADIC_SIM_ENV_PROBE=from-root\n' > "$W/root/.env"
"$W/venv/bin/dyadic-sim" --root "$W/root" run --list-cases | grep -q no_complaint || fail "--root run --list-cases"
(cd "$W/root" && env -u DYADIC_SIM_ENV_PROBE "$PY" -c "from dyadic_sim.core import paths; paths.load_env(); import os; assert os.environ.get('DYADIC_SIM_ENV_PROBE') == 'from-root'") || fail ".env from root"
echo "   --root works and <root>/.env is loaded"

echo "== bundled priors render the pinned prompts (golden hashes)"
"$PY" -m pytest -q -rs -p no:cacheprovider --import-mode=importlib --rootdir "$REPO" \
  "$REPO/tests/test_golden_prompts.py" "$REPO/tests/test_case_priors.py" "$REPO/tests/test_prior_vocabulary.py" \
  | grep -E "SKIPPED|passed|failed"
echo "PASS"
