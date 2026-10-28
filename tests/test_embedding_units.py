"""Guard the embedding input against silent truncation.

The most consequential bug this project has had. Every session-level embedding
score produced before this test existed was computed on a truncated input:
`all-MiniLM-L6-v2` accepts 256 word-piece tokens and a concatenated session
runs many times that, so every one was cut and only its opening reached the
model. Sentence Transformers truncates silently, with no warning and no error,
so nothing in the pipeline could reveal it.

The documentation described this as "dilution" over all ten turns. It was not
dilution. Everything after approximately the opening turn was discarded, and the
measure could not answer whether a symptom appeared later in a session.

These tests exist so that cannot happen again unnoticed. They do not assert that
turn-level is short enough - it often is not, 40% of turns also exceed the limit
- they assert that we know the numbers and that anything claiming whole-text
coverage actually has it.

    uv run pytest tests/test_embedding_units.py -q
"""

from __future__ import annotations

import pytest

pytest.importorskip("sentence_transformers")

from analysis.embeddings import EMBEDDINGS_AVAILABLE, get_model

# Imported, never redefined. A tested chunker that production could not call was
# the reason truncation survived the first round of fixes: the implementation
# under test and the implementation in use were different code.
from analysis.validation.sessions import chunk_by_tokens, n_tokens


@pytest.fixture(scope="module")
def model():
    if not EMBEDDINGS_AVAILABLE:
        pytest.skip("sentence-transformers not available")
    return get_model()


def test_model_limit_is_what_the_analysis_assumes(model):
    """If this changes, coverage figures computed under the old limit no longer hold."""
    assert model.max_seq_length == 256


def test_a_full_session_does_not_fit_and_we_say_so(model):
    """The premise of the finding: concatenated sessions are over the limit.

    Synthetic text at realistic session length, so the test does not depend on
    any corpus being present.
    """
    session = " ".join(
        "I have not been sleeping and everything feels heavy lately." for _ in range(150)
    )
    assert n_tokens(model, session) > model.max_seq_length, (
        "a session-length string now fits the model; the truncation finding and "
        "the --unit guidance both need revisiting"
    )


def test_chunking_keeps_every_chunk_within_the_limit(model):
    session = " ".join(
        f"Sentence number {i} about not sleeping and feeling heavy for weeks."
        for i in range(200)
    )
    chunks = chunk_by_tokens(model, session)
    assert len(chunks) > 1, "long text should split"
    oversized = [c for c in chunks if n_tokens(model, c) > model.max_seq_length]
    assert not oversized, f"{len(oversized)} chunk(s) exceed the model limit"


def test_chunking_loses_no_sentences(model):
    """Coverage, not just compliance: chunking must not drop text."""
    sentences = [f"This is sentence {i} and it is about sleep." for i in range(60)]
    chunks = chunk_by_tokens(model, " ".join(sentences))
    joined = " ".join(chunks)
    missing = [s for s in sentences if s not in joined]
    assert not missing, f"{len(missing)} sentence(s) lost during chunking"


def test_a_single_oversized_sentence_is_split_within_the_limit(model):
    """A sentence too long to fit is split, not passed through oversized.

    The earlier helper returned such a sentence whole and still over the limit,
    so one run-on turn silently reintroduced exactly the truncation these tests
    exist to prevent. Splitting happens at tokenizer boundaries and each piece
    is re-measured, because decode/encode does not reliably round-trip.
    """
    monster = "I feel " + "very " * 400 + "tired."
    chunks = chunk_by_tokens(model, monster)
    assert chunks and chunks[0].strip(), "oversized single sentence disappeared"
    assert len(chunks) > 1, "oversized sentence was not split"
    oversized = [c for c in chunks if n_tokens(model, c) > model.max_seq_length]
    assert not oversized, f"{len(oversized)} piece(s) still exceed the limit"
