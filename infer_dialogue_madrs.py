"""Same as `dyadic-sim infer-dialogue-madrs`."""
import runpy

runpy.run_module("dyadic_sim.cli.infer_dialogue_madrs", run_name="__main__", alter_sys=True)
