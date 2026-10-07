# dyadic-sim

A simulation framework for structured conversations between two AI agents in asymmetric roles, for now focussed on the therapist-patient dyad.

---

## Overview

`dyadic-sim` runs and analyses dyadic interactions between LLM-based agents. Each agent receives a set of *priors* (constitutive orientations, not scripts) that define its role, and the simulation tracks how the interaction unfolds over multiple turns.

The current focus is on the **therapist-patient dyad** because it offers a particularly well-structured test case:

- The roles are inherently asymmetric (one holds the frame, the other works within it).
- Ethical constraints (e.g., abstinence, non-exploitation) actively shape the interaction rather than merely limiting it.
- The goal of therapy is its own ending, which gives the simulation a clear trajectory to track.

The framework is designed to be adaptable. You can swap models, modify priors, add new patient cases, and plug in different analysis modules.

---

## What It Measures

The simulation measures whether symptoms written into the patient prior are actually
*expressed* in the patient agent's speech, and at what severity:

| Analysis | What it detects |
|---|---|
| Symptom relevance | Whether a given turn discusses a symptom at all, before it is scored |
| Symptom severity | Per-turn and per-session severity ratings from a pluggable rating model (`symptom_scoring/`) |
| Construct validity | Whether an injected symptom is expressed rather than silently ignored (`analysis/validation/`) |

The severity step is deliberately instrument-agnostic. `symptom_scoring/` defines the
pipeline (parse transcript → filter for relevance → rate → aggregate to session
level); everything specific to a given clinical scale is bundled in an `Instrument`
(`symptom_scoring/instrument.py`) — its topics, the patterns that mark a turn as
being about a topic, the score range, the language its rater expects, and how the
topics line up with the PHQ anchors in patient priors.

MADRS is the instrument currently wired up, as a worked example rather than a
commitment to that scale. Swapping it means defining another `Instrument` and
pointing `--instrument` at it, not editing the pipeline.

See [Validation Analyses](#validation-analyses) for how these are run.

### Research status

Everything here is **exploratory**. The pipeline was developed by looking at the
same two corpora it is now applied to, so its results evaluate feasibility,
robustness and measurement behaviour — not a preregistered hypothesis. One cell
(`afraid_of_dogs` × `sleep_problems`, n=63 per arm) separates injected from control
on two independent instruments and shows no leak into non-sleep arms; most cells do
not, and one symptom with a *larger* raw lift fails on specificity.

**No automated measure here has been validated against human judgement.** The
blinded coding protocol is written and the sampling and scoring code is dry-run, but
no human coding has been performed. Until it has, the correct phrasing is "the
lexicon fired on 17% of sleep-injected sessions", never "17% of patients reported
sleep problems". Full record: `manuscript/VALIDATION.md`.

**Generator version.** Sessions produced by this branch are `generation_version: 5`:
prior + case + symptoms only, with no hidden agenda, no hazard monitor and no per-turn
state compression, and a therapist prior that does not supply emotion words or go
looking for a problem (see `simulation/utterance.py` for the changelog). `power01` was
generated at version 3. Base rates do not carry across versions, so the first batch at
the current version is a new pilot, not the confirmatory run.

---

## Agents and Priors

Each agent receives a layered prior that establishes who they are before the conversation starts.

**Therapist priors:**

| Layer | Example |
|---|---|
| Role prior | Maintain asymmetry; hold the space without filling it |
| Ethical priors | Abstinence; non-exploitation; non-abandonment |
| Agenda prior | Facilitate insight over relief; track what is not said |
| Self prior | Tolerates not-knowing; stays curious under pressure |
| Relational prior | The patient will seek and fear recognition simultaneously |

**Patient priors:**

| Layer | Example |
|---|---|
| Presenting complaint | "I feel profoundly alone, like I exist behind glass" |
| Theory of cure | "If someone truly loved me, the emptiness would go away" |
| Relational pattern | Anxious attachment; seeks merger; tests loyalty |
| Transference expectation | "They will eventually find me too much" |
| Resistance structure | Becomes pleasing on the surface; hides real pain |

Every field in a case file is shown to the patient agent; there is no hidden layer.
The case is the presenting picture, and any depressive symptom is added per condition
by `run_symptom_experiments.py` (see Validation Analyses), so a case file as-is is the
control.

### Context

Each agent's context on every turn is its prior (the system prompt) plus the full
transcript so far, as alternating messages. Nothing else is carried between turns. An
earlier version injected a per-turn, LLM-written state summary into the system prompt;
it changed what the patient said for reasons unrelated to the injected symptom and
doubled the calls per turn, so it is gone. Follow-up sessions are not implemented yet;
the intended mechanism is one end-of-session summary per role, generated once from the
transcript and passed into the priors of the next session.

### Model Flexibility

The same simulation code runs across providers, so any combination is
*expressible*. What has actually been run is narrower:

```
Therapist                Patient
-----------------------------------------
llama3.1            x    llama3.1         # run: null condition, same model both sides
mistral-nemo        x    mistral-nemo     # run: null condition, second family
llama3.1            x    mistral-nemo     # NOT run: does not fit in 12 GB VRAM
claude-sonnet-4-6   x    gpt-4o           # deferred: cloud, agents implemented
claude-sonnet-4-6   x    llama3.1         # deferred: hybrid
```

**Every session generated so far is a same-model dyad.** The cross-model local
pair is code-complete but has never run: llama3.1 (5.5 GB resident at 4096
context) plus mistral-nemo (7.7 GB) exceeds the 11.8 GB free on the development
machine, so the two would evict and reload each other on every turn. Frontier API
models are implemented and deliberately deferred to keep the pipeline local and
reproducible; their entries in `config/models.yaml` are commented out and their
model IDs are stale. Measured figures and the third-model decision:
`manuscript/INFRASTRUCTURE.md`.

---

## Project Structure

```
dyadic-sim/
|
|-- env.example                  # API key template: copy to .env and fill in
|-- pyproject.toml               # dependencies managed by uv; pytest pythonpath
|-- run.py                       # entry point: one session
|-- run_symptom_experiments.py   # batch-generate symptom-isolated sessions
|-- evaluate_session_symptoms.py # score existing sessions with the scoring pipeline
|-- README.md                    # this file
|
|-- config/
|   |-- models.yaml              # model registry + active role assignments
|   |-- designs.yaml             # expected cells / runs / inclusion policy per arm
|   |-- analysis_spec.yaml       # frozen analysis parameters + evidence criteria
|   |-- priors/
|       |-- therapist/
|       |   |-- base.yaml        # role, structural, ethical, self, relational priors
|       |   |-- variants/
|       |       |-- cbt.yaml
|       |-- patient/
|           |-- cases/
|               |-- _template.yaml              # blank template for new cases
|               |-- afraid_of_dogs.yaml         # phobia; orthogonal to the depression domains
|               |-- feeling_off.yaml            # intended-neutral bed (it is not; see VALIDATION)
|               |-- empty_and_invisible.yaml    # moderate (narcissistic wound)
|               |-- only_love_can_save_me.yaml  # aloneness; a theory of cure the frame cannot meet
|               |-- no_complaint.yaml           # null bed: nothing wrong, a spare session
|               |-- *_run_*.yaml                # generated per-condition cases (gitignored)
|
|-- agents/
|   |-- base_agent.py            # abstract interface all providers implement
|   |-- local_agent.py           # Ollama (Llama, Mistral, Gemma, ...)
|   |-- claude_agent.py          # Anthropic Claude
|   |-- openai_agent.py          # OpenAI GPT-4o, o1
|   |-- agent_factory.py         # builds correct agent from model name
|
|-- priors/
|   |-- loader.py                # loads + validates YAML prior files
|   |-- therapist_prior.py       # therapist prior dataclass + system prompt builder
|   |-- patient_prior.py         # patient prior dataclass + system prompt builder
|
|-- simulation/
|   |-- dyad.py                  # orchestrates the two-agent exchange
|   |-- opening.py               # the therapist's first words (speech, not a stage cue)
|   |-- utterance.py             # speech-only rule + stage-direction stripper; GENERATION_VERSION
|   |-- turn_manager.py          # builds per-turn context: prior + transcript
|   |-- session.py               # session lifecycle: start / resume / close
|
|-- analysis/
|   |-- ANALYSIS.md              # the analysis layer, in detail
|   |-- embeddings.py            # shared sentence-transformer loader (revision-pinnable)
|   |-- validation/              # manipulation-validity checks (see Validation Analyses)
|       |-- paths.py             # canonical data/ locations; verifies MANIFEST
|       |-- sessions.py          # session loader: status, text units, token chunking
|       |-- quality.py           # QC flags: role-confused / annotation-only turns
|       |-- design.py            # corpus vs config/designs.yaml
|       |-- design_status.py     # which cells are done, at what n, what is missing
|       |-- provenance.py        # <script>.analysis_config.json beside every result
|       |-- spec.py              # applies config/analysis_spec.yaml verdict rule
|       |-- symptom_lexicon.py   # the naive lexicons, as editable word lists
|       |-- lexicon_detect.py    # negation-scoped detection, both speakers
|       |-- naive_prevalence.py  # PHQ-9 + MADRS panels, rate / base rate / contrast
|       |-- embed_core.py        # shared embedding core (units, controls, estimand)
|       |-- symptom_embed.py     # embedding manipulation check (target_z family)
|       |-- symptom_embed_robust.py  # reference-wording robustness, same core
|       |-- exemplars.py         # a-priori exemplar pools: purity, overlap, refsets
|       |-- exemplars/           # the pools themselves + GENERATION.md provenance
|       |-- heatmap.py           # shared condition x symptom figures
|       |-- rerun.py             # every corpus through the spec + all sensitivities
|       |-- coding_sample.py     # blinded human-coding items + sealed key
|       |-- coding_analysis.py   # coder agreement, then instrument validity
|
|-- symptom_scoring/
|   |-- prior_vocabulary.py      # PHQ-9: KEYS, LABELS (prompt), REFERENCES (analysis)
|   |-- instrument.py            # Instrument: topics, patterns, range, prior mapping
|   |-- instruments/
|   |   |-- madrs.py             # the MADRS taxonomy (the one currently wired up)
|   |-- pipeline.py              # end-to-end: transcript -> per-session symptom scores
|   |-- transcript_parser.py     # reads session transcripts into scoreable turns
|   |-- relevance_detector.py    # does this turn discuss a topic at all?
|   |-- madrs_bert_model.py      # current severity rater (swappable; MADRS-BERT)
|   |-- turn_scorer.py           # per-turn scoring
|   |-- session_aggregator.py    # per-turn scores -> session-level score
|   |-- translator.py            # translation step, only if the rater needs it
|   |-- result_writer.py         # writes scores to data/results/
|   |-- evaluation.py            # prior intent vs coverage vs expression vs rating
|   |-- types.py                 # typed records passed between stages
|   |-- config.py                # prior vocabulary, thresholds, runtime settings
|
|-- tests/                       # uv run pytest -q
|
|-- data/                        # gitignored (see Validation Analyses)
    |-- sessions/
    |   |-- {session_id}/
    |       |-- transcript.jsonl                  # full turn-by-turn exchange
    |       |-- metadata.json                     # models, priors, generation_version
    |-- external/                                 # imported raw batches + MANIFEST.yaml
    |-- results/                                  # everything derived
    |-- _archive/                                 # frozen artifacts, not regenerable
```

`manuscript/` holds the research record and is gitignored: `VALIDATION.md` (what
the analyses found), `ANALYSIS_SPEC.md` (the frozen spec), `CODING_PROTOCOL.md`,
`CODE_FINDINGS.md` (engineering debt), `INFRASTRUCTURE.md` (measured hardware
constraints). `analysis/ANALYSIS.md` is gitignored on the same grounds. Links to
either below will only resolve in a working copy.

---

## Quick Start

### 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:
```bash
uv --version
```

### 2. Clone and install dependencies

```bash
git clone https://github.com/your-username/dyadic-sim.git
cd dyadic-sim
uv sync
```

That's it: `uv sync` reads `pyproject.toml`, creates a virtual environment, and installs everything. No manual `pip install`, no conda, no activation needed.

### 3. Set up your API keys

```bash
cp env.example .env
```

Open `.env` and fill in what you have. For local-only piloting you don't need any API keys:

```bash
# .env

# Leave these blank if you're running locally only
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# Ollama runs on localhost (no key needed)
OLLAMA_BASE_URL=http://localhost:11434

DEFAULT_THERAPIST_MODEL=llama3.1
DEFAULT_PATIENT_MODEL=llama3.1
```

### 4. Install Ollama and pull local models

**Install Ollama:**
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

**Start the Ollama server:**
```bash
ollama serve
```

**Pull models** (do this once, they're stored locally):
```bash
# Recommended for piloting (fits in 12GB VRAM)
ollama pull llama3.1        # 8B, best all-round small model
ollama pull mistral-nemo    # 12B, noticeably more capable, fits tight
```

### 5. Configure your models

Open `config/models.yaml` and set the `roles` section:

```yaml
roles:
  therapist: llama3.1
  patient: llama3.1
```

### 6. Run your first session

```bash
# Fully local, zero cost, good for piloting
uv run python run.py \
  --therapist llama3.1 \
  --patient llama3.1 \
  --case afraid_of_dogs \
  --orientation cbt \
  --turns 10

# A second model family (same model both sides — see Model Flexibility)
uv run python run.py \
  --therapist mistral-nemo \
  --patient mistral-nemo \
  --case empty_and_invisible \
  --turns 20

# Continue an existing session for more turns
uv run python run.py \
  --resume data/sessions/session_001 \
  --additional-turns 10
```

### 7. Add your own patient cases

```bash
cp config/priors/patient/cases/_template.yaml \
   config/priors/patient/cases/my_new_case.yaml
```

Open the file and fill in each field. Everything in it is shown to the patient agent,
so write it in the patient's own voice and keep symptom vocabulary out of it
(`tests/test_case_priors.py` checks this). Symptoms are added per condition by
`run_symptom_experiments.py`, never written into the case. Case headers say only what
the case is; the design notes and revision history for every case live in
`manuscript/CASES.md` (research record, gitignored).

## Piloting Strategy

For now, we run local models only:

```
Stage 1: null condition, one family (done)
  uv run python run.py --therapist llama3.1 --patient llama3.1 --case afraid_of_dogs
  Goal: does the role structure do anything at all with the simplest case?

Stage 2: null condition, a second family (done — all current results)
  uv run python run.py --therapist mistral-nemo --patient mistral-nemo --case afraid_of_dogs
  Goal: does the finding hold on a different pretraining lineage?

Stage 3: genuine alterity — different models in each role (NOT yet run)
  Goal: does a real other change the dynamics?
  Blocked: the llama3.1 x mistral-nemo pair does not fit in 12 GB VRAM, so it
  would evict and reload on every turn. Needs a smaller partner (~6 GB) or a
  bigger card. See manuscript/INFRASTRUCTURE.md.
```

### Tests

```bash
uv run pytest -q          # pythonpath is configured in pyproject.toml
```

---

## Validation Analyses

`analysis/validation/` holds checks on the
**construct validity of experimental manipulations** — currently the symptom-injection
study, which asks whether a PHQ-9 depression symptom written into the patient prior is
actually *expressed* by the patient agent (rather than silently ignored).

Data lives under `data/`, split first by **raw vs derived** — local vs external
applies only to raw. All of it is gitignored (`external/` and `_archive/` are
typically symlinks to external storage):

```
data/
├── sessions/                    RAW, generated on this machine
│   └── session_*/
├── external/                    RAW, imported from elsewhere (read-only)
│   ├── MANIFEST.yaml            origin + fingerprint per batch
│   ├── _archives/               the zips each batch came from
│   ├── llama31-2026-05/         150 sessions
│   └── mistral-nemo-2026-06/    150 sessions
└── results/                     DERIVED, everything computed here
    ├── <experiment>/            + <script>.analysis_config.json per artifact
    └── _logs/                   batch-generation console logs, one per arm
```

Sessions killed mid-write land in `data/sessions/` with `turn_count: 0`.
`sessions.py` classifies those as `incomplete` and excludes them by default with
a stated reason, so they never reach an analysis — no quarantine directory is
needed.

Every raw session sits exactly one level below its bucket, so a single glob
shape reads any batch — `analysis/validation/paths.py` derives the lookup from
the manifest rather than hardcoding directory layouts.

**Adding an external batch.** Drop it in as `data/external/<name>/session_*/`,
then add an entry to `MANIFEST.yaml`. Directory names are readable and safe to
rename; the `model` field is the analysis label and appears as the `model`
column in output CSVs, so changing it breaks comparability with frozen
baselines. Record a `fingerprint` — a stable hash over sorted
`session_id:sha256(transcript)` lines — to detect duplicate imports and to check
a batch has not changed since the analysis that cites it:

```bash
PYTHONPATH=. python -c "from analysis.validation.paths import fingerprint; \
  print(fingerprint('data/external/<name>'))"
PYTHONPATH=. python -c "from analysis.validation.paths import verify; \
  print(verify())"          # check every batch against its recorded value
```

Paths are centralised in `analysis/validation/paths.py`; run the scripts from the repo
root with `PYTHONPATH=.`. Defaults point at the active corpus declared there
(`ACTIVE_PREFIX`, currently `power01`, read from local `data/sessions/`) and write
into that corpus's results directory; pass `--sessions` and `--prefix` together for
anything else, so one corpus's results are never overwritten with another's.

**Which instrument to believe is not a free choice.** The lexicon is primary
because every firing reads back to the words that caused it; the embedding is
convergent evidence only; MADRS-BERT is an external check; blinded human coding is
the validation target and **does not exist yet**. The ranking, the pinned
parameters and the rule for what counts as evidence are frozen in
`config/analysis_spec.yaml`. `analysis/ANALYSIS.md` is the detailed map of this
package.

**1. Generate symptom-isolated sessions.** With `--injection all_nine` (the
default) one PHQ-9 symptom is set to a frequency anchor and the other eight read
"not at all"; with `--injection active_only` the eight are omitted entirely, so
the prompt never names them:

```bash
uv run python run_symptom_experiments.py \
  --therapist mistral-nemo --patient mistral-nemo \
  --case afraid_of_dogs --injection active_only \
  --symptoms depressed_mood,psychomotor_changes,sleep_problems \
  --frequency "nearly every day" --repeats 20 --prefix pilot1
```

Sessions land in `data/sessions/` tagged `pilot1_<case>_<symptom>_run_<n>`. A
matched `no_symptoms` control is generated automatically — do not list it in
`--symptoms`. Keep `--injection` and both models fixed across every top-up at one
prefix; each changes generation, so mixing them silently splits the arm. Deepen a
cell with `--start-run` (the number `design_status.py` prints) and `--no-control`
so existing controls are not regenerated.

**2. Check coverage** before analysing — derived from the sessions on disk, not
from a hand-kept ledger, and compared against what `config/designs.yaml` declares:

```bash
PYTHONPATH=. python analysis/validation/design_status.py --prefix pilot1 --target 20
```

**3. Run the lexicon panel (primary).** Condition × symptom prevalence for the
PHQ-9 and MADRS lexicons, patient expression and therapist uptake, each raw and
against a matched control. `--evidence` dumps every firing with the word that
matched and the sentence it came from, so any cell can be audited by hand:

```bash
PYTHONPATH=. python analysis/validation/naive_prevalence.py \
  --sessions 'data/sessions/*' --prefix pilot1 --clean-text --evidence \
  --out-dir data/results/pilot1
```

Read the **control row first**: a lexicon that fires on most sessions regardless of
what was injected tells you nothing about the diagonal above it.

**4. Run the embedding check (convergent).** Cosine similarity of the patient's
speech to each PHQ-9 reference, z-scored within session, plus the between-session
contrast against the matched control (`target_z_vs_control` — read this one):

```bash
PYTHONPATH=. python analysis/validation/symptom_embed.py \
  --sessions 'data/sessions/*' --prefix pilot1 \
  --unit bounded-turn --unit-agg max \
  --references expanded --reference-agg mean \
  --clean-text --by-case --out-dir data/results/pilot1
```

⚠ **`--unit session` is truncated, always.** The embedding model accepts 256
word-piece tokens and Sentence Transformers truncates silently; every whole-session
input overflows it, so `session` describes roughly the opening turn and exists for
reproduction only. `bounded-turn` splits just the turns that overflow and treats an
over-limit input as a hard error. Each of the flags above is known to move results
and all of them are recorded in the provenance sidecar written beside the output.

Also in the package: `symptom_embed_robust.py` (varies the reference wording and
nothing else), `exemplars.py` (audits the a-priori exemplar pools), and `rerun.py`
(every corpus through the frozen spec plus all declared sensitivities, each varying
exactly one parameter). `symptom_embed.py` and `naive_prevalence.py` both take
`--by-case` and `--qc-view`; flagged content is reported, never silently dropped.

### Session symptom scoring

`evaluate_session_symptoms.py` produces research-only symptom severity scores from
saved sessions. The pipeline is fixed; the rating model is not. It pairs each patient
response with the preceding therapist message, filters turn-topic combinations for
relevance, hands the accepted pairs to a rating model, and aggregates to one score
per topic per session.

The instrument currently wired up is MADRS (`--instrument madrs`, the default), rated
by `webesama/MADRS-BERT`. That checkpoint declares `language="de"`, so accepted English
pairs are translated first; an instrument whose rater already works in the transcript
language skips translation entirely. The per-topic maximum accepted score is kept.

Prior anchors (PHQ) and the resulting severity scores are reported separately and must
not be interpreted as the same scale. Each instrument declares how well its topics
approximate the PHQ anchors (`close` / `partial` / `approximate` / `none`), and that
quality flag is carried into every result.

⚠ **Read the relevance gate, not the severity score.** On this corpus MADRS-BERT's
severity head is uncalibrated: wherever relevance fires it returns 3.5–5.4 out of 6
across every topic, controls included. Each topic also carries
`number_of_relevant_turns`, and *that* is the usable output — detection rate agrees
with the lexicon where the two overlap. The pipeline also reads `patient_text` raw
(there is no `--clean-text`), so on transcripts generated before the speech-only fix
it would score stage directions as patient speech.

**Adding an instrument.** Define an `Instrument` in `symptom_scoring/instruments/` and
register it in that package's `REGISTRY`:

```python
MY_SCALE = Instrument(
    key="my_scale",
    name="MyScale",
    topics=("mood", "sleep"),
    topic_descriptions={"mood": "...", "sleep": "..."},   # English, for relevance
    relevance_patterns={"mood": r"\b(sad|down)\b", ...},  # English, pre-translation
    score_range=(0.0, 3.0),
    language="en",                                        # what the rater expects
    prior_topic_map={"depressed_mood": "mood", "psychomotor_changes": None},
    prior_mapping_quality={"depressed_mood": "close", "psychomotor_changes": "none"},
)
```

It is checked for consistency on construction (every topic described and matchable,
prior map pointing only at real topics), and the pipeline needs no changes. A rater
for a new instrument only has to provide `prepare_input` and `predict_batch`.

Score one session in PowerShell:

```powershell
uv run python .\evaluate_session_symptoms.py `
  --session-path .\data\sessions\session_001 `
  --experiment-name pilot1 `
  --output-dir .\data\results\symptom_scores
```

Score all saved sessions:

```powershell
uv run python .\evaluate_session_symptoms.py `
  --sessions-dir .\data\sessions `
  --experiment-name pilot1 `
  --output-dir .\data\results\symptom_scores
```

This creates:

```text
data/results/symptom_scores/pilot1/
  <session_id>/
    symptom_scores.json
    turn_scores.csv
  sessions_summary.csv
  errors.jsonl  # only when a session fails
```

Use a different `--experiment-name` for each experiment. Names may contain
letters, numbers, dots, underscores, and hyphens. If the option is omitted,
the per-session folders are written directly under `--output-dir`.

With the default rater, the first run downloads MADRS-BERT and the German
translation checkpoint it depends on. Add
`--device cpu` to force CPU execution or `--relevance-mode rules` to use only
transparent relevance rules without the semantic-similarity fallback.

### Patient-dialogue MADRS fine-tuning

The repository also includes a separate experiment for predicting item-level
MADRS severity from patient speech alone. It builds one symptom-aware input per
session/item (`SYMPTOM: <item> [SEP] <patient dialogue>`), splits whole patients
or sessions to prevent leakage, fine-tunes the same MADRS-BERT regressor with
early stopping and checkpoints, and reports overall and per-item metrics. The
existing session symptom-scoring pipeline above is unchanged. See
[`DIALOGUE_MADRS.md`](DIALOGUE_MADRS.md) for the data contract and commands.

---

## Theoretical Background

For the full psychological and philosophical motivation behind this project, see
[`manuscript/README.md`](manuscript/README.md) — the manuscript directory is
gitignored, so that link resolves in a working copy only.

An earlier version of the analysis layer operationalised this background directly,
as six "personhood markers" (semantic drift, reciprocal determination, role-self
tension, Aufhebung structure, recognition dynamics, telos tracking). They were
removed when the project narrowed to symptom scoring and now live only on the
branch `archive/personhood-markers`; see `analysis/ANALYSIS.md`.

## License

BSD 3-Clause License: see `LICENSE`.

## AI Disclosure

AI tools (Claude, Codex) were used during development to audit code, fix bugs, and brainstorm possible implementations.
