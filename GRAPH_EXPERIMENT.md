# Graph-first PHQ-9 round-trip experiment

`run_graph_experiment.py` adds a separate experimental arm. It does not change
the established symptom-injection batches or their analyses.

## Design

The default run uses the existing `no_complaint` case in ten conditions: one
without a PHQ-9 symptom and one for each of the nine symptoms at **nearly every
day**. It makes five graph construction attempts per condition, two graph-derived
profiles per graph, and one ten-turn session for each profile: 50 graphs, 100
profiles, and 100 sessions. Identical graphs across deterministic attempts are
retained and reported, not silently replaced. The patient agent sees its
profile, never the source prior, injected symptom block, or graph. The therapist
keeps the current CBT prior. Both dialogue roles use Mistral-Nemo.

Forward graph construction adapts MP1, MP2, MP3, and the model-based MP5
ensemble from [Paper 1's OSF supplement](https://osf.io/download/mx8ya/).
The necessary change is that prior sentences and the PHQ-9 item replace
transcript utterances. Reverse graph construction uses patient turns from the
saved transcript. Profiles use the intake-form fields and graph-grounding rules
from [Graph2Counsel's released prompts](https://github.com/UKPLab/emnlp2026-graph2counsel/tree/main/prompts_system_user).
Two-profile output and fidelity to `no_complaint` are study-specific adaptations.

The construction models are Llama-3.1-70B for process and theme stages, then
Llama-3.1-70B, Qwen-2.5-72B, and GPT-4o-mini with majority voting for links.
GPT-4o creates and reviews profiles and compares round trips. These are the
papers' model choices in `--model-mode paper`. An explicitly requested
`--model-mode local` keeps the Llama, Qwen, and Mistral-Nemo stages, replacing
GPT-4o-mini with `gpt-oss:20b` and GPT-4o with `gemma3:27b` by default.
`LOCAL_EDGE_MODEL_ID` and `LOCAL_PROFILE_MODEL_ID` can select other local
Ollama models such as `gpt-oss:120b` for the profile stage. The local variant
must be analyzed separately from a paper-model run. Graph2Counsel's
one-shot dialogue generator is replaced by this repository's alternating
Mistral-Nemo dyad, as required by the experiment.

## Setup and run

Follow the Windows [access and configuration steps](README.md#get-access-and-configure-it-on-windows)
in the README. This script reads `.env` directly; `config/models.yaml` does not
choose its graph models. The two large open models may run on any compatible
server. Without `GRAPH_MODEL_BASE_URL`, they are sought in local Ollama. Set
`OPENAI_API_KEY` for GPT-4o and GPT-4o-mini and run `mistral-nemo` in Ollama.
If the graph server exposes different IDs for the same weights, set
`PAPER_LLAMA_MODEL_ID` and `PAPER_QWEN_MODEL_ID`; both IDs are recorded in the
run manifest.

```powershell
uv run python .\run_graph_experiment.py --check
uv run python .\run_graph_experiment.py --conditions no_symptoms --repeats 1 --output-dir .\data\graph_experiments\pilot_access
uv run python .\run_graph_experiment.py
```

For a fully local run after pulling the models listed in the
[README](README.md#run-without-paid-model-apis-on-another-laptop):

```powershell
uv run python .\run_graph_experiment.py --model-mode local --check
uv run python .\run_graph_experiment.py --model-mode local --conditions no_symptoms --repeats 1 --output-dir .\data\graph_experiments\pilot_local
uv run python .\run_graph_experiment.py --model-mode local --output-dir .\data\graph_experiments\no_complaint_phq9_local_v1
```

The preflight checks all required model IDs before creating any run artifacts.
Run the pilot in its own directory: a design manifest prevents reusing that
directory for the full experiment.
The default output is `data/graph_experiments/no_complaint_phq9_graph_v1/`.
Rerunning the same command resumes incomplete sessions and skips completed
stages. Changing the declared design requires a new output directory.

Each condition/run folder contains `source_input.json`, `source_graph.json`,
`source_graph.svg`, `profiles.json`, and `llm_trace.jsonl` with prompts,
responses, model IDs, and temperatures. Each profile folder contains its
profile, a pointer to the existing session transcript, reconstructed graph and
SVG, reconstructed profile, and `comparison.json`. `summary.csv` indexes
per-session node and edge recall, profile coverage, symptom recovery, and
contradictions. A null denominator is written as an empty CSV cell.

The comparison is descriptive: links inferred from a short session are not
verified causal relationships, and ten turns may not reveal every profile fact.
