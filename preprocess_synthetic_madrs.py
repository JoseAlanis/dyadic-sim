"""Same as `dyadic-sim preprocess-synthetic-madrs`."""
import runpy

runpy.run_module("dyadic_sim.cli.preprocess_synthetic_madrs", run_name="__main__", alter_sys=True)
