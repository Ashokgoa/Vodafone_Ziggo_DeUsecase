"""Tests for app.retrieval.retrieve.

Uses hand-picked vectors and a real (temporary) Chroma collection, the
same approach as test_vector_store.py: precise, deterministic distances
without depending on a real embedding model.
"""

from app.retrieval.vector_store import add_chunks, get_collection
from app.retrieval.retrieve import retrieve


def test_retrieve_is_confident_when_a_close_match_exists(tmp_path) -> None:
    collection = get_collection(str(tmp_path))
    add_chunks(collection, ["relevant chunk"], [[1.0, 0.0]], source_url="https://example.com")

    # [0.99, 0.01] points almost exactly the same direction as [1.0, 0.0],
    # so its cosine distance is very small (a close match).
    result = retrieve(
        query_embedding=[0.99, 0.01], collection=collection, top_k=3, max_distance=0.1
    )

    assert result.is_confident is True
    assert result.chunks[0]["text"] == "relevant chunk"


def test_retrieve_is_not_confident_when_the_best_match_is_too_far(tmp_path) -> None:
    collection = get_collection(str(tmp_path))
    add_chunks(collection, ["unrelated chunk"], [[1.0, 0.0]], source_url="https://example.com")

    # [0.0, 1.0] is perpendicular to [1.0, 0.0]: a large cosine distance,
    # simulating a question with no real match in the store.
    result = retrieve(
        query_embedding=[0.0, 1.0], collection=collection, top_k=3, max_distance=0.1
    )

    assert result.is_confident is False
    assert result.chunks == []


def test_retrieve_is_not_confident_when_the_store_is_empty(tmp_path) -> None:
    collection = get_collection(str(tmp_path))  # nothing added

    result = retrieve(
        query_embedding=[1.0, 0.0], collection=collection, top_k=3, max_distance=0.5
    )

    assert result.is_confident is False
    assert result.chunks == []
