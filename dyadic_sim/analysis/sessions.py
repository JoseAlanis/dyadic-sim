"""
Shared session loading for the symptom analyses.

Centralises the logic that was duplicated across symptom_embed.py,
symptom_embed_robust.py and the since-deleted symptom_manifest.py: parsing the
generated case name,
extracting patient text (dropping the verbatim-echo artifact), and iterating
session directories from an arbitrary glob.

`parse_case` handles ANY run prefix, including multi-word ones, not just the
hardcoded "batch_" the original scripts assumed.
"""
from __future__ import annotations

import glob
import json
import os
import re

from dyadic_sim.core import paths
from dyadic_sim.generation.utterance import clean_utterance


def _discover_cases() -> tuple[str, ...]:
    """Hand-written case names, read from the cases directory.

    This was a hardcoded tuple of the three cases that existed when it was
    written, so `feeling_off` parsed to (prefix, None, None, run) and every
    session using it was silently dropped by `iter_sessions` and therefore by
    every analysis built on it. Discovering the names instead means a new case
    works the moment its file exists.

    Generated per-condition files are excluded by their "_run_" marker, the
    same rule .gitignore uses, so this is stable even while a batch is running
    and the directory is full of them. Longest first, so a case name that
    contains another still matches correctly.
    """
    found = []
    for directory in paths.config_dirs("priors", "patient", "cases"):
        for entry in os.listdir(directory):
            name, extension = os.path.splitext(entry)
            if extension != ".yaml" or "_run_" in name or name.startswith("_"):
                continue
            found.append(name)
    if not found:
        # Directory missing (e.g. analysing an exported corpus elsewhere).
        found = ["only_love_can_save_me", "empty_and_invisible", "afraid_of_dogs"]
    return tuple(sorted(dict.fromkeys(found), key=len, reverse=True))


CASES = _discover_cases()

# <prefix>_<case>_<symptom-or-no_symptoms>_run_<n>   ; prefix may contain underscores
_RUN_RE = re.compile(r"^(?P<body>.+)_run_(?P<run>\d+)$")


def parse_case(case_name: str):
    """Return (prefix, case, symptom, run) from a generated case_name.

    symptom is "no_symptoms" for the baseline condition. Returns
    (None, None, None, None) if the name doesn't match the expected pattern.

    The case marker is located FIRST and the prefix is whatever precedes it. A
    non-greedy `(?P<prefix>.+?)_` captured only up to the first underscore, so
    `pilot_fix_afraid_of_dogs_sleep_problems_run_1` parsed its prefix as
    "pilot" - meaning `--prefix pilot_fix` silently matched nothing, and two
    experiments sharing a first component would have been merged. Longest case
    name wins, so a case whose name is a prefix of another still resolves.
    """
    m = _RUN_RE.match(case_name)
    if not m:
        return None, None, None, None
    body, run = m.group("body"), int(m.group("run"))
    for c in sorted(CASES, key=len, reverse=True):
        idx = body.find(c)
        if idx == -1:
            continue
        prefix = body[:idx].rstrip("_") or None
        tail = body[idx + len(c):].lstrip("_")
        return prefix, c, (tail or "no_symptoms"), run
    # No known case marker: fall back to the old split so malformed names still
    # yield a prefix rather than nothing.
    head, _, _ = body.partition("_")
    return head or None, None, None, run


def patient_turns(session_dir: str, clean: bool = False) -> list[tuple[int, str]]:
    """Patient utterances as (turn_index, text), same filtering as `patient_text`.

    Exists because a concatenated session EXCEEDS THE EMBEDDING MODEL'S INPUT
    LIMIT and is silently truncated. all-MiniLM-L6-v2 accepts 256 word-piece
    tokens; sessions run a median of ~2,300, so 100% are cut and only ~11% of
    each is seen. The earlier explanation here said the symptom was "diluted"
    into the session average - it was not averaged, it was discarded, and
    everything after roughly the opening turn never reached the model.

    Turns are far closer to the limit (median ~235 tokens) but 40% still exceed
    it, so callers wanting guaranteed coverage should chunk. Verified by
    tests/test_embedding_units.py.
    """
    path = os.path.join(session_dir, "transcript.jsonl")
    with open(path) as handle:
        tx = [json.loads(l) for l in handle if l.strip()]
    turns = []
    for i, t in enumerate(tx):
        p = (t.get("patient_text") or "").strip()
        if i > 0 and p and p == (tx[i - 1].get("therapist_text") or "").strip():
            continue
        if p:
            turns.append((i, clean_utterance(p) if clean else p))
    return turns


# --------------------------------------------------------------------------
# Text units for embedding.
#
# `all-MiniLM-L6-v2` accepts 256 word-piece tokens and Sentence Transformers
# truncates anything longer SILENTLY - no warning, no error. Every session-level
# score ever produced was computed on a truncated input. These helpers exist so
# that any analysis claiming to have read a whole text actually has.
#
# The chunker lived in tests/test_embedding_units.py, which meant the tested
# implementation was not the one production could use. It lives here now and the
# test imports it.
# --------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Reserve room for [CLS] and [SEP], which the model adds after we count.
_SPECIAL_TOKENS = 2


def n_tokens(model, text: str) -> int:
    return len(model.tokenizer.encode(text, add_special_tokens=False))


def _split_oversized_sentence(model, sentence: str, limit: int) -> list[str]:
    """Split one sentence that cannot fit, at tokenizer boundaries.

    The previous implementation returned such a sentence whole and still over
    the limit, so a single run-on turn silently reintroduced truncation. Decode
    is not guaranteed to round-trip to the same token count (word-piece
    fragments can re-tokenize differently), so each piece is re-measured and the
    window shrinks until it genuinely fits rather than being assumed to.
    """
    ids = model.tokenizer.encode(sentence, add_special_tokens=False)
    pieces: list[str] = []
    start, window = 0, limit
    while start < len(ids):
        piece = model.tokenizer.decode(ids[start:start + window])
        while window > 1 and n_tokens(model, piece) > limit:
            window = max(1, window * 3 // 4)
            piece = model.tokenizer.decode(ids[start:start + window])
        pieces.append(piece)
        start += window
    return pieces or [sentence]


def chunk_by_tokens(model, text: str, limit: int | None = None) -> list[str]:
    """Sentence-packed chunks that never exceed the model's token limit.

    Packing by sentence rather than by token index keeps chunks readable, which
    matters because every hit has to be auditable against the transcript. A
    sentence too long to fit on its own is split at tokenizer boundaries rather
    than passed through oversized.
    """
    limit = (limit or model.max_seq_length) - _SPECIAL_TOKENS
    out: list[str] = []
    current: list[str] = []
    total = 0
    for raw in (s for s in _SENTENCE_SPLIT.split(text) if s.strip()):
        for sentence in (
            [raw] if n_tokens(model, raw) <= limit
            else _split_oversized_sentence(model, raw, limit)
        ):
            size = n_tokens(model, sentence)
            if total + size > limit and current:
                out.append(" ".join(current))
                current, total = [], 0
            current.append(sentence)
            total += size
    if current:
        out.append(" ".join(current))
    return out or [text]


#: Units that guarantee every token reaches the model. `session` and `turn` do
#: not, and are retained only to reproduce historical numbers.
BOUNDED_UNITS = ("bounded-turn", "chunk")
UNITS = ("session", "turn", *BOUNDED_UNITS)


def text_units(session_dir: str, *, unit: str = "turn", clean: bool = False,
               model=None) -> list[tuple[int, str]]:
    """Patient text as (turn_index, text) pairs, under the chosen unit.

    `turn_index` is preserved through splitting, so a chunk or fragment can
    still be read back to the turn it came from. `session` uses -1, having no
    single turn of origin.

      session       one string, TRUNCATED at 256 tokens. Reproduction only.
      turn          one unit per turn. Historical default; a long turn still
                    truncates.
      bounded-turn  as `turn`, but splits only the turns that overflow, so it
                    stays comparable to historical turn scoring while
                    guaranteeing coverage.
      chunk         sentence-packed chunks over all patient text, ignoring turn
                    boundaries.
    """
    if unit not in UNITS:
        raise ValueError(f"unknown unit {unit!r}; expected one of {UNITS}")
    if unit == "session":
        return [(-1, patient_text(session_dir, clean=clean))]

    turns = patient_turns(session_dir, clean=clean)
    if unit == "turn":
        return turns
    if model is None:
        raise ValueError(f"unit={unit!r} needs a model to count tokens")

    limit = model.max_seq_length - _SPECIAL_TOKENS
    if unit == "bounded-turn":
        out: list[tuple[int, str]] = []
        for index, text in turns:
            if n_tokens(model, text) <= limit:
                out.append((index, text))
            else:
                out.extend((index, piece) for piece in chunk_by_tokens(model, text))
        return out

    # chunk: one stream, so attribute each chunk to the turn it starts in.
    out = []
    for index, text in turns:
        out.extend((index, piece) for piece in chunk_by_tokens(model, text))
    return out


def patient_text(session_dir: str, clean: bool = False) -> str:
    """Concatenate patient utterances, dropping verbatim echoes of the prior
    therapist turn (the echo data artifact).

    `clean=True` also strips stage directions and leaked role labels. This
    changes results, and unevenly: some models narrate constantly and others
    almost never, so a cross-model comparison on raw text is partly a comparison
    of narration style. It inflates `psychomotor_changes` in particular, because
    gesture words - "leaning", "fidgeting", "pauses" - are literally psychomotor
    descriptions being read as psychomotor symptoms. Off by default so that a
    result computed on raw text stays reproducible; turn it on for new runs.
    """
    path = os.path.join(session_dir, "transcript.jsonl")
    with open(path) as handle:
        tx = [json.loads(l) for l in handle if l.strip()]
    chunks = []
    for i, t in enumerate(tx):
        p = (t.get("patient_text") or "").strip()
        if i > 0 and p and p == (tx[i - 1].get("therapist_text") or "").strip():
            continue
        if p:
            chunks.append(clean_utterance(p) if clean else p)
    return " ".join(chunks)


# --------------------------------------------------------------------------
# Session status.
#
# Four outcomes, not two. Conflating them is why `design_status.py` twice
# reported a cell as full while it held a 5-turn fragment, and why a partial
# transcript could be ingested by a broad glob despite the status warning.
#
# The distinction that matters most: a session stopped by the former hazard
# protocol (generation_version <= 3 corpora; the generator no longer pauses) is
# a DESIGNED outcome, not a failure. It is shorter than the others, which biases
# a max-over-units score downward, so it must be visible and separately
# excludable - but excluding it by default would silently drop the very sessions
# where the patient said something serious enough to trigger a pause.
# --------------------------------------------------------------------------

COMPLETE = "complete"
CRISIS_TERMINATED = "crisis_terminated"
INCOMPLETE = "incomplete"
MALFORMED = "malformed"

#: Analysed unless a caller says otherwise. Crisis terminations are in, because
#: they are real sessions; accidental partials and malformed directories are out.
DEFAULT_STATUSES = (COMPLETE, CRISIS_TERMINATED)


def classify_session(session_dir: str, meta: dict) -> tuple[str, str | None]:
    """(status, reason). `turn_count` is 0 exactly while a session is unfinished.

    `turn_count` is written as 0 when a session starts and set to the real total
    when it finishes, so it distinguishes "still being written or killed" from
    "ended early on purpose" - which transcript length alone cannot.
    """
    if not os.path.exists(os.path.join(session_dir, "transcript.jsonl")):
        return MALFORMED, "no transcript.jsonl"
    if not meta.get("case_name"):
        return MALFORMED, "no case_name in metadata"
    turns = meta.get("turn_count", 0)
    if not turns:
        return INCOMPLETE, "turn_count is 0: interrupted or still being written"
    if meta.get("paused_at_turn"):
        return CRISIS_TERMINATED, (
            f"hazard protocol paused at turn {meta['paused_at_turn']}"
            f" ({meta.get('reason', 'unspecified')})"
        )
    return COMPLETE, None


def scan_sessions(globs) -> list[dict]:
    """Every matched directory with its status, INCLUDING the excluded ones.

    Separate from `iter_sessions` so that what was left out is reportable rather
    than invisible.
    """
    if isinstance(globs, str):
        globs = [globs]
    seen, out = set(), []
    for g in globs:
        for d in sorted(glob.glob(g)):
            if d in seen or not os.path.isdir(d):
                continue
            seen.add(d)
            mpath = os.path.join(d, "metadata.json")
            if not os.path.exists(mpath):
                out.append({"dir": d, "meta": {}, "prefix": None, "case": None,
                            "symptom": None, "run": None, "status": MALFORMED,
                            "reason": "no metadata.json"})
                continue
            try:
                with open(mpath) as handle:
                    meta = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                out.append({"dir": d, "meta": {}, "prefix": None, "case": None,
                            "symptom": None, "run": None, "status": MALFORMED,
                            "reason": f"unreadable metadata.json: {exc}"})
                continue
            prefix, case, symptom, run = parse_case(meta.get("case_name", ""))
            status, reason = classify_session(d, meta)
            if case is None or symptom is None:
                status, reason = MALFORMED, "case_name does not parse"
            out.append({"dir": d, "meta": meta, "prefix": prefix, "case": case,
                        "symptom": symptom, "run": run,
                        "status": status, "reason": reason})
    return out


def iter_sessions(globs, statuses=DEFAULT_STATUSES, excluded=None):
    """Yield (session_dir, metadata, prefix, case, symptom, run) for analysable
    sessions matched by `globs`.

    Accidental partials and malformed directories are excluded BY DEFAULT: the
    old behaviour let a broad glob ingest a half-written transcript even while
    `design_status.py` was warning about it. Pass `statuses` to widen or narrow
    the selection, and `excluded` (a list) to collect what was skipped and why.

    `meta` carries `_status` and `_status_reason` so downstream code can record
    them per row without re-deriving the classification.
    """
    for entry in scan_sessions(globs):
        if entry["status"] not in statuses:
            if excluded is not None:
                excluded.append(entry)
            continue
        meta = dict(entry["meta"])
        meta["_status"] = entry["status"]
        meta["_status_reason"] = entry["reason"]
        yield (entry["dir"], meta, entry["prefix"], entry["case"],
               entry["symptom"], entry["run"])