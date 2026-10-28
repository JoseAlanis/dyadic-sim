"""Same as `dyadic-sim run-symptom-experiments`."""
import runpy

runpy.run_module("dyadic_sim.cli.run_symptom_experiments", run_name="__main__", alter_sys=True)
