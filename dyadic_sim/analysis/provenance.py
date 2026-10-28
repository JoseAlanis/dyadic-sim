"""Write a configuration sidecar beside every result artifact.

A result CSV used to record none of the choices that produced
it. Whether text was cleaned, whether the unit was session or turn, single or
expanded references, which aggregation, which model revision, which lexicon —
none of it was stored. Some appeared only in figure titles, which are not
machine-readable provenance, and the embedding model revision was unpinned, so a
rerun could silently produce different numbers under the same filenames.

That matters more here than usual, because several of those choices have been
shown to move results: cleaning shifts `psychomotor_changes` by 0.51, session
versus turn changes the contrast by 0.3, and `max` versus `mean` aggregation
flips it by 0.4. A result without its configuration cannot be compared to
anything.

Hashes rather than copies for the lexicon and references: the point is to detect
that they changed, not to duplicate them, and both live in version control.

Two failures of the first version are fixed
here.

**One filename per script, not one per directory.** `symptom_embed.py` and
`naive_prevalence.py` both wrote `analysis_config.json` into the same output
directory, so the second run silently replaced the first. It cost real evidence
in both directions - one corpus kept only its embedding configuration, another
only its lexicon one - and which survived depended on nothing but run order, so
a headline result can end up with no record of how it was produced. Sidecars
are now named `<script-stem>.analysis_config.json`.

**Fingerprint the inputs, not their count.** The old sidecar recorded
`{"n_sessions": 189}`, which cannot establish *which* sessions produced a number.
`corpus_fingerprint()` now records sorted session IDs and a SHA-256 of each
`metadata.json` and `transcript.jsonl`, plus one aggregate hash over all of them,
so a rerun on a changed or differently-filtered corpus is detectable rather than
merely suspected.

**Two records, answering different questions.** The sidecar answers "how were
the files sitting in this directory produced" and describes only the current
artifacts. `runs.jsonl` beside it is append-only and answers "what has this
directory been" — every run that has written here, including the ones since
replaced.

That split is what lets a results directory be overwritten freely. Writing used
to REFUSE when the incoming configuration was incompatible with the existing
sidecar, because overwriting destroyed the only record of the current artifacts.
The cost was a new directory per fix — `_v2`, `_predfix`, `_final` — names that
stop meaning anything the moment the thing they were distinguished from is
archived, and that leave stale numbers sitting under current-looking names.

Now the superseded configuration is appended to the log *before* it is replaced,
so an overwrite costs the artifacts but not the record, and the refusal is a
printed note instead. This is the right trade while a pipeline is still being
refined and every fix would otherwise fork the directory tree. It is NOT the
right trade for a frozen result: archive that, and the log travels with it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5, check=False)
        sha = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, timeout=5, check=False).stdout.strip()
        # "-dirty" is the important part: nearly every result in this project so
        # far was produced from an uncommitted tree, and saying so is honest.
        return f"{sha}{'-dirty' if dirty else ''}" if sha else None
    except (OSError, subprocess.SubprocessError):
        return None


def lexicon_fingerprint() -> dict:
    """Hash the lexicon term lists and the reference/exemplar pools."""
    from dyadic_sim.analysis import exemplars
    from dyadic_sim.analysis.symptom_lexicon import HAZARD_TIERS, MADRS, PHQ9
    from dyadic_sim.core.vocabulary import LABELS, REFERENCES

    def lex_repr(d):
        return json.dumps(
            {k: [list(v.self_framed), list(v.symptom_terms), list(v.health_terms)]
             for k, v in d.items()},
            sort_keys=True,
        )

    pools = {}
    for stem in exemplars.available():
        records = exemplars.load_pool(stem)
        pools[stem] = {"n": len(records),
                       "sha": _sha256(json.dumps([r.get("text") for r in records],
                                                 sort_keys=True))}
    return {
        "phq9_lexicon": _sha256(lex_repr(PHQ9)),
        "madrs_lexicon": _sha256(lex_repr(MADRS)),
        "hazard_tiers": _sha256(lex_repr(HAZARD_TIERS)),
        "prompt_labels": _sha256(json.dumps(LABELS, sort_keys=True)),
        "single_references": _sha256(json.dumps(REFERENCES, sort_keys=True)),
        "exemplar_pools": pools,
    }


def embedding_fingerprint() -> dict:
    try:
        from dyadic_sim.analysis.embeddings import EMBEDDINGS_AVAILABLE, get_model
    except ImportError:
        return {"available": False}
    if not EMBEDDINGS_AVAILABLE:
        return {"available": False}
    model = get_model()
    inner = getattr(model, "_first_module", lambda: None)()
    auto = getattr(inner, "auto_model", None)
    return {
        "available": True,
        "name": getattr(model, "model_card_data", None)
        and getattr(model.model_card_data, "base_model", None) or "all-MiniLM-L6-v2",
        # The limit that silently truncated every session-level result. Recorded
        # so a future rerun under a different limit is detectable.
        "max_seq_length": getattr(model, "max_seq_length", None),
        "revision": getattr(getattr(auto, "config", None), "_commit_hash", None),
    }


def _file_sha(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def corpus_fingerprint(session_dirs) -> dict:
    """Identify the exact inputs, not merely how many there were.

    A count cannot answer "which sessions produced this number", which is the
    question that matters when a corpus grows, a filter changes, or a partial
    session is quarantined. Each session contributes the hash of its
    `metadata.json` and `transcript.jsonl`; the aggregate is a hash over all of
    them in sorted order, so any addition, removal or edit changes it.
    """
    per = {}
    for directory in sorted(session_dirs):
        session_id = os.path.basename(directory.rstrip(os.sep))
        per[session_id] = {
            "metadata": _file_sha(os.path.join(directory, "metadata.json")),
            "transcript": _file_sha(os.path.join(directory, "transcript.jsonl")),
        }
    aggregate = _sha256(json.dumps(per, sort_keys=True))
    return {
        "n_sessions": len(per),
        "session_ids": sorted(per),
        "aggregate_sha": aggregate,
        "per_session_sha": per,
    }


def _package_versions() -> dict:
    out = {}
    for name in ("sentence_transformers", "transformers", "torch", "numpy",
                 "pandas", "scipy", "matplotlib"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            out[name] = None
    return out


# Fields whose disagreement means two runs are not comparable, so overwriting
# one with the other would destroy evidence rather than refresh it. Everything
# else (timestamps, platform, argv order) may differ freely.
_COMPARABILITY_KEYS = ("lexicon", "embedding_model", "corpus", "args")


def is_compatible(old: dict, new: dict) -> tuple[bool, list[str]]:
    """Would overwriting `old` with `new` lose information?"""
    differing = [k for k in _COMPARABILITY_KEYS if old.get(k) != new.get(k)]
    return (not differing), differing


def sidecar_name(script: str) -> str:
    """`symptom_embed.py` -> `symptom_embed.analysis_config.json`."""
    return f"{os.path.splitext(os.path.basename(script))[0]}.analysis_config.json"


#: Append-only history of every run that has written into a results directory.
#: Lives beside the artifacts so it travels with them when the directory is
#: archived or moved.
RUN_LOG = "runs.jsonl"


def append_run_log(out_dir: str, payload: dict, *, superseded: bool = False) -> str:
    """Record one run in `<out_dir>/runs.jsonl`, never rewriting what is there.

    This is what makes overwriting a results directory safe. The refusal below
    exists because replacing a sidecar destroyed the only record of how the
    current artifacts were produced; the answer to that is not to forbid the
    overwrite but to stop it being lossy. A superseded configuration is appended
    here before it is replaced, so the history survives even though only the
    latest artifacts do.

    Deliberately append-only and deliberately not the sidecar: the sidecar
    answers "how were the files sitting here produced", the log answers "what
    has this directory been". Both are needed and they are different questions.
    """
    entry = dict(payload)
    entry["superseded"] = superseded
    entry["logged_at"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
    path = os.path.join(out_dir, RUN_LOG)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
    return path


def write(out_dir: str, *, script: str, args, inputs: dict | None = None,
          session_dirs=None, extra: dict | None = None) -> str:
    """Write `<script>.analysis_config.json` into `out_dir` and return its path.

    Always writes. An incompatible existing configuration is appended to
    `runs.jsonl` first and reported, rather than blocking the run.
    """
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "script": os.path.basename(script),
        "written_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
        "git_commit": _git_commit(),
        "argv": sys.argv[1:],
        "args": {k: v for k, v in sorted(vars(args).items())} if args else {},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        "inputs": inputs or {},
        "lexicon": lexicon_fingerprint(),
        "embedding_model": embedding_fingerprint(),
    }
    if session_dirs is not None:
        payload["corpus"] = corpus_fingerprint(session_dirs)
    if extra:
        payload.update(extra)

    path = os.path.join(out_dir, sidecar_name(script))
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                old = json.load(handle)
        except (OSError, json.JSONDecodeError):
            old = None
        if old is not None:
            ok, differing = is_compatible(old, payload)
            if not ok:
                # The configuration about to be replaced is preserved first, so
                # the overwrite costs the ARTIFACTS but not the RECORD. This
                # used to raise instead, which forced a new directory per fix
                # and cluttered results/ with `_v2`, `_predfix` and the like -
                # names that stop meaning anything the moment the thing they
                # were distinguished from is archived.
                append_run_log(out_dir, old, superseded=True)
                print(f"  note: replacing artifacts in {out_dir} produced under a "
                      f"different configuration (differs in: {', '.join(differing)}). "
                      f"The superseded configuration is appended to {RUN_LOG}.")
    append_run_log(out_dir, payload)
    with open(path, "w", encoding="utf-8") as handle:
        # `default=str` because callers pass arbitrary extras (a loaded YAML
        # spec contains real date objects, for instance). Provenance failing
        # to serialise must never be able to kill an analysis that has
        # already done its work.
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    return path
