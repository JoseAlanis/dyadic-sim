"""Same as `dyadic-sim run`."""
import runpy

runpy.run_module("dyadic_sim.cli.run", run_name="__main__", alter_sys=True)
