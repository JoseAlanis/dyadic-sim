"""Same as `dyadic-sim evaluate-session-symptoms`."""
import runpy

runpy.run_module("dyadic_sim.cli.evaluate_session_symptoms", run_name="__main__", alter_sys=True)
