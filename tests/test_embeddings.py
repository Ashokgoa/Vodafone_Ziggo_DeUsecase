"""Tests for app.retrieval.embeddings.

These mock SentenceTransformer itself, so tests never download or load the
real model — they stay fast and work offline.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.retrieval.embeddings import _get_model, embed_texts


@pytest.fixture(autouse=True)
def clear_model_cache():
    # _get_model caches loaded models by name. Clearing it before and after
    # each test stops one test's mock model from leaking into the next.
    _get_model.cache_clear()
    yield
    _get_model.cache_clear()


@patch("app.retrieval.embeddings.SentenceTransformer")
def test_embed_texts_returns_one_vector_per_text(mock_st_class: MagicMock) -> None:
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
    mock_st_class.return_value = mock_model

    vectors = embed_texts(["hello", "world"], model_name="fake-model")

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    mock_model.encode.assert_called_once_with(["hello", "world"], convert_to_numpy=True)


@patch("app.retrieval.embeddings.SentenceTransformer")
def test_embed_texts_loads_the_model_only_once(mock_st_class: MagicMock) -> None:
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1]])
    mock_st_class.return_value = mock_model

    embed_texts(["a"], model_name="fake-model")
    embed_texts(["b"], model_name="fake-model")

    # The model should be constructed once and reused across calls (the
    # lru_cache in _get_model), not reloaded from disk every time.
    mock_st_class.assert_called_once()
