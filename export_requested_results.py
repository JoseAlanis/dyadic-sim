"""Same as `dyadic-sim export-requested-results`."""
import runpy

runpy.run_module("dyadic_sim.cli.export_requested_results", run_name="__main__", alter_sys=True)
