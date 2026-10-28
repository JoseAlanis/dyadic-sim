# dyadic-sim

Simulated therapy sessions between two LLM agents, a therapist and a patient.
Each agent is given a written prior, and the two take turns over a shared
transcript. A patient prior can carry a PHQ-9 symptom at a stated frequency, so
sessions can be generated with a known difference between conditions. Saved
sessions can then be scored for symptom expression.

dyadic-sim is a research instrument. It does not provide therapy and is not a
clinical tool.

## About

| | |
|---|---|
| CI | [![tests](https://github.com/JoseAlanis/dyadic-sim/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/JoseAlanis/dyadic-sim/actions/workflows/tests.yml) |
| Meta | [![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE) [![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](pyproject.toml) |

## How a session works

- **Priors.** The therapist prior is `config/priors/therapist/base.yaml` plus an
  orientation variant (`variants/cbt.yaml`). A patient prior is a case file in
  `config/priors/patient/cases/`, written in the patient's own voice; a symptom
  block can be added per condition.
- **Turns.** The therapist opens with a fixed line. On each turn the patient
  answers and the therapist replies. Each agent receives its prior as the system
  prompt and the conversation so far, in the order it was spoken.
- **Speech only.** Both priors ask for spoken words only, and every recorded turn
  is cleaned of stage directions, role labels and wrapping quotes. When cleaning
  changes a turn, the original text is kept alongside.
- **Record.** Each session is written to `data/sessions/<session_id>/`:
  `transcript.jsonl` (one row per turn: the patient line and the therapist
  reply), `metadata.json` (models, case, injected symptom levels, opening line,
  `generation_version`) and a PDF transcript. A session can only be continued by
  the generator version that started it.

## Install

Requires Python 3.13, [uv](https://docs.astral.sh/uv/) and
[Ollama](https://ollama.com/).

```bash
git clone https://github.com/JoseAlanis/dyadic-sim.git
cd dyadic-sim
uv sync
ollama pull mistral-nemo
```

## Models

`config/models.yaml` lists the models per provider and the default role
assignments. Only Ollama is enabled, so sessions stay on the machine. Agents for
Anthropic and OpenAI exist in `agents/`; to use one, uncomment its provider
block and put the API key in `.env` (see `env.example`). Transcripts are then
sent to that provider. Use one provider per experiment.

## Run

```bash
# one session
uv run python run.py --therapist mistral-nemo --patient mistral-nemo \
  --case no_complaint --orientation cbt --turns 10
uv run python run.py --list-cases
uv run python run.py --resume data/sessions/<session_id> --additional-turns 5

# a symptom-injection batch: per run, one control and one session per symptom
uv run python run_symptom_experiments.py --therapist mistral-nemo --patient mistral-nemo \
  --case no_complaint --injection active_only --repeats 10 --prefix <prefix>
```

Batch sessions are named `<prefix>_<case>_<symptom>_run_<n>`, which is how the
analysis tools group them. `--symptoms`, `--frequency`, `--start-run`,
`--no-control` and `--control-only` are described in `--help`.

## Add a patient case

Copy `config/priors/patient/cases/_template.yaml` to `cases/<name>.yaml` and fill
in the five fields in the patient's own words. Leave symptoms out; the batch
runner adds them per condition. `uv run pytest tests/test_case_priors.py` checks
that a case contains no symptom vocabulary.

## Score sessions

```bash
uv run python evaluate_session_symptoms.py --sessions-dir data/sessions \
  --output-dir data/results/symptom_scores
```

Each patient turn that discusses a MADRS topic is rated by the MADRS-BERT
checkpoint (after translation to German, the language the rater expects), and
ratings are aggregated per session. Pin checkpoints with `--model-revision` and
`--translator-revision`; unpinned runs record the revision they resolved.

## Analyse a batch

`dyadic_sim/analysis/` compares injected conditions with controls within a
batch: a negation-aware symptom lexicon, an embedding measure, and checks that a
batch matches its declared design. Declare designs in `config/designs.yaml`
(start from `config/designs.example.yaml`). Each script documents its use under
`--help`.

Clinician rating packages for saved sessions are in `study_web/` (see its
README).

## Experiments

- `GRAPH_EXPERIMENT.md`: graph-first PHQ-9 round trip (process graph, intake
  profile, session voiced from the profile, reconstruction).
- `DIALOGUE_MADRS.md`: fine-tuning MADRS-BERT on patient-only dialogue.

## Tests

```bash
uv run pytest -q
uv run ruff check .
```

`uv run pytest --cov` adds a coverage table per file; `uv run coverage html`
then writes a browsable report to `htmlcov/`.
`tests/packaging/check_installed_wheel.sh` builds the wheel, installs it into a
fresh environment and uses it from outside the repository.

## License

BSD 3-Clause License: see `LICENSE`.

## AI Disclosure

AI tools (Claude, Codex) were used during development to audit code, fix bugs, and brainstorm possible implementations.
