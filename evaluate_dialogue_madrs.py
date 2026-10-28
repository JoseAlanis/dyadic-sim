"""Same as `dyadic-sim evaluate-dialogue-madrs`."""
import runpy

runpy.run_module("dyadic_sim.cli.evaluate_dialogue_madrs", run_name="__main__", alter_sys=True)
