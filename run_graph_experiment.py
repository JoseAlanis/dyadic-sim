"""Same as `dyadic-sim run-graph-experiment`."""
import runpy

runpy.run_module("dyadic_sim.cli.run_graph_experiment", run_name="__main__", alter_sys=True)
