"""Same as `dyadic-sim train-dialogue-madrs`."""
import runpy

runpy.run_module("dyadic_sim.cli.train_dialogue_madrs", run_name="__main__", alter_sys=True)
