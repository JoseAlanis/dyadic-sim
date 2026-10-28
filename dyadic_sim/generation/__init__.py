"""Simulation package.

Exports are resolved lazily. `generation.dyad` imports `generation.priors`, and
the priors import `generation.utterance` for the shared speech-only instruction, so an
eager re-export here would make that a circular import: loading `priors` would
trigger this module, which would load `dyad`, which needs `priors` back before
it has finished initialising.
"""

__all__ = ["Dyad", "Session", "new_session", "resume_session"]


def __getattr__(name: str):
    if name == "Dyad":
        from dyadic_sim.generation.dyad import Dyad

        return Dyad
    if name in {"Session", "new_session", "resume_session"}:
        from dyadic_sim.generation import session

        return getattr(session, name)
    raise AttributeError(name)
