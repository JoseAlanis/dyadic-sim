"""Shared sentence embedding model loader.

`DYADIC_SIM_EMBEDDING_REVISION` pins the checkpoint to a commit SHA; leave it
unset to track whatever the model id currently points at.
"""

import os

EMBEDDING_MODEL_ID = os.getenv("DYADIC_SIM_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_MODEL_REVISION = os.getenv("DYADIC_SIM_EMBEDDING_REVISION") or None

try:
    from sentence_transformers import SentenceTransformer
    _MODEL = None

    def get_model() -> SentenceTransformer:
        global _MODEL
        if _MODEL is None:
            _MODEL = SentenceTransformer(
                EMBEDDING_MODEL_ID,
                revision=EMBEDDING_MODEL_REVISION,
            )
        return _MODEL

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

    def get_model():
        raise ImportError("sentence-transformers is required for embedding analysis")
