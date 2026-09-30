"""Tests for the FastAPI app's lifespan: auto-ingestion when the store is empty.

Mocks every dependency lifespan touches (settings, the vector store,
ingestion, and graph building), so entering/exiting the app's lifespan
here never touches a real vector store, embedding model, or LLM.
"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.main import app


@patch("app.api.main.build_graph")
@patch("app.api.main.run_ingestion")
@patch("app.api.main.get_collection")
@patch("app.api.main.get_settings")
def test_lifespan_ingests_automatically_when_store_is_empty(
    mock_get_settings: MagicMock,
    mock_get_collection: MagicMock,
    mock_run_ingestion: MagicMock,
    mock_build_graph: MagicMock,
) -> None:
    mock_get_settings.return_value = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_get_collection.return_value = mock_collection
    mock_build_graph.return_value = "the-compiled-graph"

    # Using TestClient as a context manager is what actually triggers the
    # app's lifespan startup/shutdown events.
    with TestClient(app):
        pass

    mock_run_ingestion.assert_called_once()


@patch("app.api.main.build_graph")
@patch("app.api.main.run_ingestion")
@patch("app.api.main.get_collection")
@patch("app.api.main.get_settings")
def test_lifespan_skips_ingestion_when_store_already_has_chunks(
    mock_get_settings: MagicMock,
    mock_get_collection: MagicMock,
    mock_run_ingestion: MagicMock,
    mock_build_graph: MagicMock,
) -> None:
    mock_get_settings.return_value = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 25  # a persisted volume from a prior run
    mock_get_collection.return_value = mock_collection
    mock_build_graph.return_value = "the-compiled-graph"

    with TestClient(app):
        pass

    # Re-scraping and re-embedding on every restart would be wasteful (and
    # slow); a non-empty store means a previous run already did this.
    mock_run_ingestion.assert_not_called()
