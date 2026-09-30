"""Persist chunk embeddings to a local vector store, and search them.

This is the sixth stage of the ingestion pipeline, and is reused as the
first step of retrieval at question-answering time:

    ... chunk -> embed -> STORE -> RETRIEVE -> ...

We use Chroma, a small embedded vector database (no separate server to
run) that stores each chunk's text, embedding, and metadata together and
persists them to a folder on disk. That combination is what satisfies the
assignment's "local vector store" requirement without us hand-rolling our
own similarity-search math and text/vector bookkeeping.

Note: we always pass embeddings we've already computed (via
app.retrieval.embeddings.embed_texts) into add_chunks() and search().
Chroma can compute its own embeddings internally if you don't supply any,
using a default downloaded model -- we deliberately avoid that path so
there is exactly one embedding model used everywhere in this project.
"""

from __future__ import annotations

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

# This project only ever indexes one page at a time, so a single fixed
# collection name is enough -- no need to make it configurable yet.
_COLLECTION_NAME = "ziggo_chunks"


def get_collection(persist_directory: str) -> Collection:
    """Open (or create) the on-disk collection that stores our chunks.

    PersistentClient writes its index under persist_directory and reloads
    it automatically the next time the same path is used -- this is what
    makes the store "local" and durable across separate runs of ingestion
    or the API.
    """
    # anonymized_telemetry=False stops Chroma from making an outbound
    # network call on startup to report anonymous usage stats. It isn't a
    # secret leak, but this project should work fully offline once the
    # embedding model is cached, and that background call was observed to
    # hang for a long time on a restricted network -- disabling it made
    # collection creation instant and reliable.
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        # Cosine similarity (the angle between two vectors, ignoring their
        # length) is the standard choice for sentence-embedding models;
        # Chroma's default is squared L2 distance instead.
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(
    collection: Collection,
    chunks: list[str],
    embeddings: list[list[float]],
    source_url: str,
) -> None:
    """Store each chunk's text and embedding, tagged with its source URL.

    Re-running ingestion for the same source_url replaces its old chunks
    instead of piling up duplicates alongside them, so ingestion is safe
    to run again after the page content changes.
    """
    collection.delete(where={"source_url": source_url})

    ids = [f"{source_url}::{i}" for i in range(len(chunks))]
    metadatas = [{"source_url": source_url, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def search(
    collection: Collection, query_embedding: list[float], top_k: int
) -> list[dict]:
    """Find the top_k stored chunks closest to query_embedding.

    Returns plain dicts (text, distance, metadata), ordered from most to
    least relevant, so callers don't need to know Chroma's own result
    format (which nests everything in single-element lists, since the
    underlying API supports querying with several vectors at once).
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return [
        {"text": doc, "distance": dist, "metadata": meta}
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        )
    ]
