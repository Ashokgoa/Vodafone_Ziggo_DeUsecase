"""Tests for app.retrieval.vector_store.

Uses a real (but temporary, throwaway) Chroma collection rather than a
mock: Chroma runs locally with no network calls and no model download, so
unlike the heavier embedding model in test_embeddings.py there's nothing
slow or external to avoid here. pytest's tmp_path fixture gives each test
its own disposable directory, so tests never touch the project's real
data/vectorstore folder.
"""

from app.retrieval.vector_store import add_chunks, get_collection, search


def test_search_returns_the_most_similar_chunk_first(tmp_path) -> None:
    collection = get_collection(str(tmp_path))

    # Hand-picked 2D vectors instead of real embeddings: chunk 0 points in
    # nearly the same direction as the query, chunk 1 points perpendicular
    # to it. This lets the test check our storage/search logic without
    # depending on a real embedding model.
    chunks = ["Ziggo biedt internet met 500 Mbit/s.", "De hond rent in het park."]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    add_chunks(collection, chunks, embeddings, source_url="https://example.com/page")

    results = search(collection, query_embedding=[0.9, 0.1], top_k=1)

    assert len(results) == 1
    assert results[0]["text"] == chunks[0]
    assert results[0]["metadata"]["source_url"] == "https://example.com/page"
    assert results[0]["metadata"]["chunk_index"] == 0


def test_add_chunks_replaces_previous_chunks_for_the_same_source(tmp_path) -> None:
    collection = get_collection(str(tmp_path))

    add_chunks(collection, ["oude tekst"], [[1.0, 0.0]], source_url="https://example.com/page")
    add_chunks(collection, ["nieuwe tekst"], [[1.0, 0.0]], source_url="https://example.com/page")

    results = search(collection, query_embedding=[1.0, 0.0], top_k=10)

    # Re-running ingestion for the same URL should leave only the latest
    # chunks behind, not accumulate both old and new ones.
    assert [r["text"] for r in results] == ["nieuwe tekst"]
