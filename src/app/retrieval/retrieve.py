"""Retrieve relevant chunks for a question, with low-confidence handling.

This is the retrieval stage of the pipeline:

    ... store -> RETRIEVE -> LangGraph workflow -> LLM -> answer

Retrieval alone isn't enough: the vector store always returns its closest
matches, even when none of them are actually relevant (see the pricing
example in the README/Step 7 notes -- the page has no price text at all,
so the "closest" chunks are still a poor match). This module adds the
assignment's required "low-confidence / empty retrieval" handling: it
decides whether the best match is trustworthy enough to answer from.

retrieve() takes an already-computed query embedding rather than a raw
question string, so it stays a small, pure function to test (see
test_retrieve.py) -- callers first call app.retrieval.embeddings.embed_texts()
themselves, the same way they already do for ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass

from chromadb.api.models.Collection import Collection

from app.retrieval.vector_store import search


@dataclass(frozen=True)
class RetrievalResult:
    """The outcome of a retrieval attempt.

    is_confident is False whenever there is no chunk worth answering from
    -- either the store returned nothing, or its best match's distance was
    still above max_distance. Callers (the LangGraph workflow, in the next
    step) branch on this single flag rather than re-checking distances
    themselves.
    """

    chunks: list[dict]
    is_confident: bool


def retrieve(
    query_embedding: list[float],
    collection: Collection,
    top_k: int,
    max_distance: float,
) -> RetrievalResult:
    """Search the vector store and filter out matches that aren't close enough.

    A match only makes it into the result if its distance is at or below
    max_distance. This naturally covers both failure cases the assignment
    calls out: an empty store returns no matches at all, and a populated
    but irrelevant store returns matches that all get filtered out here.
    """
    matches = search(collection, query_embedding, top_k)
    confident_matches = [m for m in matches if m["distance"] <= max_distance]
    return RetrievalResult(chunks=confident_matches, is_confident=bool(confident_matches))
