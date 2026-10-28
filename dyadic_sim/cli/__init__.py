"""`dyadic-sim <command> [args]`: one entry point for every script.

Each command runs a module exactly as `python -m <module>` would, with the
remaining arguments. `--root DIR` (or `DYADIC_SIM_ROOT`) sets the working
directory first; user configuration (`config/`) and outputs (`data/`) are
resolved from there.
"""

from __future__ import annotations

import os
import runpy
import sys

COMMANDS = {
    "run": "dyadic_sim.cli.run",
    "run-symptom-experiments": "dyadic_sim.cli.run_symptom_experiments",
    "run-graph-experiment": "dyadic_sim.cli.run_graph_experiment",
    "evaluate-session-symptoms": "dyadic_sim.cli.evaluate_session_symptoms",
    "export-requested-results": "dyadic_sim.cli.export_requested_results",
    "preprocess-dialogue-madrs": "dyadic_sim.cli.preprocess_dialogue_madrs",
    "preprocess-synthetic-madrs": "dyadic_sim.cli.preprocess_synthetic_madrs",
    "train-dialogue-madrs": "dyadic_sim.cli.train_dialogue_madrs",
    "evaluate-dialogue-madrs": "dyadic_sim.cli.evaluate_dialogue_madrs",
    "infer-dialogue-madrs": "dyadic_sim.cli.infer_dialogue_madrs",
    "naive-prevalence": "dyadic_sim.analysis.naive_prevalence",
    "symptom-embed": "dyadic_sim.analysis.symptom_embed",
    "symptom-embed-robust": "dyadic_sim.analysis.symptom_embed_robust",
    "rerun": "dyadic_sim.analysis.rerun",
    "design-status": "dyadic_sim.analysis.design_status",
    "exemplars": "dyadic_sim.analysis.exemplars",
    "coding-sample": "dyadic_sim.analysis.coding_sample",
    "coding-analysis": "dyadic_sim.analysis.coding_analysis",
    "verify-external": "dyadic_sim.core.paths",
    "select-sessions": "dyadic_sim.study_web.select_sessions",
    "build-package": "dyadic_sim.study_web.build_package",
    "import-returns": "dyadic_sim.study_web.import_returns",
}

USAGE = "usage: dyadic-sim [--root DIR] <command> [args]\n\ncommands:\n" + "\n".join(
    f"  {name:28s} python -m {module}" for name, module in COMMANDS.items()
)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    root = os.environ.get("DYADIC_SIM_ROOT")
    if args[:1] == ["--root"] and len(args) > 1:
        root, args = args[1], args[2:]
    if not args or args[0] in {"-h", "--help"}:
        print(USAGE)
        raise SystemExit(0 if args else 2)
    command, rest = args[0], args[1:]
    if command not in COMMANDS:
        print(f"dyadic-sim: unknown command {command!r}\n\n{USAGE}", file=sys.stderr)
        raise SystemExit(2)
    if root:
        os.chdir(root)
    sys.argv = [f"dyadic-sim {command}", *rest]
    runpy.run_module(COMMANDS[command], run_name="__main__", alter_sys=True)
