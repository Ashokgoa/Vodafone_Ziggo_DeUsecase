"""Turn text into embedding vectors using a local, multilingual model.

This is the fifth stage of the ingestion pipeline (and is reused at query
time to embed the customer's question):

    scrape HTML -> extract -> clean -> chunk -> EMBED -> store -> retrieve

We run the embedding model locally (sentence-transformers) rather than
calling a cloud embeddings API: it needs no API key/secret, has no
per-call cost, and keeps ingestion fully self-contained. The trade-off is
a heavier local dependency (this pulls in torch) and a one-time model
download the first time it runs.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=2)
def _get_model(model_name: str) -> SentenceTransformer:
    """Load a model by name and cache it.

    Unlike get_settings(), we deliberately DO cache here: loading model
    weights from disk takes real time (and the first run also downloads
    them), while the model object itself holds no per-call state that
    would make reusing it incorrect. maxsize=2 is generous headroom in
    case a test or a future step ever needs two different model names in
    the same process; in normal use only one model name is ever requested.
    """
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    """Embed a list of texts, returning one vector (list of floats) per text.

    Encoding as a batch (one model.encode() call for all texts) rather than
    one call per text is both simpler and meaningfully faster, since the
    model can process multiple texts together.
    """
    model = _get_model(model_name)
    vectors = model.encode(texts, convert_to_numpy=True)
    return vectors.tolist()
