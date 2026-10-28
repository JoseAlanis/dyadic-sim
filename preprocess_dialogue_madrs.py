"""Same as `dyadic-sim preprocess-dialogue-madrs`."""
import runpy

runpy.run_module("dyadic_sim.cli.preprocess_dialogue_madrs", run_name="__main__", alter_sys=True)
