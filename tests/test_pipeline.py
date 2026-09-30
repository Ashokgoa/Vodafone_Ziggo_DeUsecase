"""Tests for app.ingestion.pipeline.

Mocks the network call (fetch_html) and the embedding model (embed_texts),
the same way the individual stage tests already do, so this never touches
the real Ziggo page or downloads a model.
"""

from unittest.mock import patch

from app.config.settings import Settings
from app.ingestion.pipeline import run_ingestion
from app.retrieval.vector_store import get_collection, search


def _fake_settings(vector_store_dir: str) -> Settings:
    return Settings(
        app_env="test",
        log_level="INFO",
        target_url="https://example.com/page",
        chunk_size=500,
        chunk_overlap=50,
        embedding_model="fake-model",
        vector_store_dir=vector_store_dir,
        retrieval_top_k=3,
        retrieval_max_distance=0.65,
        fallback_message="Sorry, no answer found.",
        llm_model="fake-llm",
        llm_max_tokens=100,
    )


@patch("app.ingestion.pipeline.embed_texts")
@patch("app.ingestion.pipeline.fetch_html")
def test_run_ingestion_stores_chunks_from_the_page(
    mock_fetch_html, mock_embed_texts, tmp_path
) -> None:
    mock_fetch_html.return_value = "<html><body><p>Ziggo internet is snel.</p></body></html>"
    mock_embed_texts.return_value = [[1.0, 0.0]]

    settings = _fake_settings(str(tmp_path))
    chunk_count = run_ingestion(settings)

    assert chunk_count == 1
    collection = get_collection(str(tmp_path))
    assert collection.count() == 1
    results = search(collection, query_embedding=[1.0, 0.0], top_k=1)
    assert "Ziggo internet is snel" in results[0]["text"]
