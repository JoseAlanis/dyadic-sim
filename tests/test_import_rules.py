"""Which layer of `dyadic_sim` may import which.

    core         nothing above it
    generation   core
    scoring      core
    analysis     core, generation.utterance, scoring.transcript_parser
    study_web    core, analysis.sessions
    experiments  core, generation, scoring, analysis
    cli          anything
Nothing outside `cli` imports `cli` or `experiments`.

`KNOWN` lists the imports that break these rules today. It may only shrink:
the test fails on any violation not listed, and on any listed entry that no
longer occurs, so a fixed exception has to be removed from the list.
"""

from __future__ import annotations

import ast
from pathlib import Path

import dyadic_sim

PACKAGE = Path(dyadic_sim.__file__).resolve().parent
ALLOWED = {
    "core": ("dyadic_sim.core",),
    "generation": ("dyadic_sim.core",),
    "scoring": ("dyadic_sim.core",),
    "analysis": ("dyadic_sim.core", "dyadic_sim.generation.utterance",
                 "dyadic_sim.scoring.transcript_parser"),
    "study_web": ("dyadic_sim.core", "dyadic_sim.analysis.sessions"),
    "experiments": ("dyadic_sim.core", "dyadic_sim.generation", "dyadic_sim.scoring",
                    "dyadic_sim.analysis"),
}

KNOWN: set[tuple[str, str]] = {
    # MADRS topic lexicon used by the naive panel
    ("dyadic_sim.analysis.naive_prevalence", "dyadic_sim.scoring.instruments.madrs"),
    # PHQ-9 frequency values, shared with scoring
    ("dyadic_sim.generation.priors.patient_prior", "dyadic_sim.scoring.config"),
    # the shared sentence-transformer loader
    ("dyadic_sim.scoring.relevance_detector", "dyadic_sim.analysis.embeddings"),
    # the therapist's opening line, needed to rebuild the first turn
    ("dyadic_sim.scoring.transcript_parser", "dyadic_sim.generation.opening"),
}


def _module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _absolute(source: str, is_package: bool, node: ast.ImportFrom) -> str:
    """The module an ImportFrom names, with relative imports resolved."""
    if node.level == 0:
        return node.module or ""
    package = source.split(".") if is_package else source.split(".")[:-1]
    base = package[: len(package) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def _targets(node: ast.AST, source: str = "", is_package: bool = False) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if isinstance(node, ast.ImportFrom):
        module = _absolute(source, is_package, node)
        # `from dyadic_sim.core import paths` imports the module dyadic_sim.core.paths
        out = []
        for alias in node.names:
            candidate = f"{module}.{alias.name}"
            base = PACKAGE.parent.joinpath(*candidate.split("."))
            out.append(candidate if base.with_suffix(".py").exists() or base.is_dir() else module)
        return out
    return []


def _allowed(source: str, target: str) -> bool:
    src_layer = source.split(".")[1] if source.count(".") else ""
    tgt_layer = target.split(".")[1] if target.count(".") else ""
    if src_layer in {"", "cli"} or src_layer == tgt_layer:
        return True
    if tgt_layer in {"cli", "experiments"}:
        return False
    return any(target == a or target.startswith(a + ".") for a in ALLOWED.get(src_layer, ()))


def violations() -> set[tuple[str, str]]:
    found = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        source = _module_name(path)
        is_package = path.name == "__init__.py"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            for target in _targets(node, source, is_package):
                if target.startswith("dyadic_sim.") and not _allowed(source, target):
                    found.add((source, target))
    return found


def test_no_new_layer_violations():
    assert violations() - KNOWN == set()


def test_known_exceptions_still_occur():
    assert KNOWN - violations() == set()


def test_relative_imports_are_resolved_and_checked():
    node = ast.parse("from ..analysis import embeddings").body[0]
    targets = _targets(node, "dyadic_sim.generation.dyad")
    assert targets == ["dyadic_sim.analysis.embeddings"]
    assert not _allowed("dyadic_sim.generation.dyad", targets[0])
    package_node = ast.parse("from . import session").body[0]
    assert _targets(package_node, "dyadic_sim.generation", is_package=True) == [
        "dyadic_sim.generation.session"]
