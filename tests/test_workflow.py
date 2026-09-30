"""Tests for app.graph.workflow.

Mocks embed_texts and ask_llm (so no real embedding model or LLM call is
needed) and uses a real, temporary Chroma collection preloaded with
hand-picked vectors (as in test_vector_store.py), so the whole graph --
embedding, retrieval, the confidence branch, and both possible endings --
runs for real and fast, without any network call or API cost.
"""

from unittest.mock import patch

from app.graph.workflow import build_graph
from app.retrieval.vector_store import add_chunks, get_collection

_BUILD_GRAPH_DEFAULTS = dict(
    embedding_model="fake-embedding-model",
    llm_model="fake-llm-model",
    llm_max_tokens=100,
)


@patch("app.graph.workflow.ask_llm")
@patch("app.graph.workflow.embed_texts")
def test_graph_generates_an_answer_when_confident(
    mock_embed_texts, mock_ask_llm, tmp_path
) -> None:
    mock_embed_texts.return_value = [[1.0, 0.0]]  # stands in for the question's embedding
    mock_ask_llm.return_value = "Het snelste abonnement is 2 Gbit/s."
    collection = get_collection(str(tmp_path))
    add_chunks(collection, ["relevant chunk"], [[1.0, 0.0]], source_url="https://example.com")

    graph = build_graph(
        collection,
        top_k=3,
        max_distance=0.1,
        fallback_message="Sorry, no answer found.",
        **_BUILD_GRAPH_DEFAULTS,
    )

    result = graph.invoke({"question": "irrelevant text, embed_texts is mocked"})

    assert result["is_confident"] is True
    assert result["answer"] == "Het snelste abonnement is 2 Gbit/s."
    # The LLM should only ever be called on the confident path, with the
    # retrieved chunk actually passed through as context.
    mock_ask_llm.assert_called_once()
    args, _ = mock_ask_llm.call_args
    assert "relevant chunk" in args[1]


@patch("app.graph.workflow.ask_llm")
@patch("app.graph.workflow.embed_texts")
def test_graph_falls_back_when_not_confident(
    mock_embed_texts, mock_ask_llm, tmp_path
) -> None:
    mock_embed_texts.return_value = [[0.0, 1.0]]  # points away from the stored chunk
    collection = get_collection(str(tmp_path))
    add_chunks(collection, ["unrelated chunk"], [[1.0, 0.0]], source_url="https://example.com")

    graph = build_graph(
        collection,
        top_k=3,
        max_distance=0.1,
        fallback_message="Sorry, no answer found.",
        **_BUILD_GRAPH_DEFAULTS,
    )

    result = graph.invoke({"question": "irrelevant text, embed_texts is mocked"})

    assert result["is_confident"] is False
    assert result["answer"] == "Sorry, no answer found."
    # The whole point of the confidence check: never spend an LLM call on
    # a question we can't ground in anything relevant.
    mock_ask_llm.assert_not_called()


@patch("app.graph.workflow.ask_llm")
@patch("app.graph.workflow.embed_texts")
def test_graph_falls_back_when_store_is_empty(
    mock_embed_texts, mock_ask_llm, tmp_path
) -> None:
    mock_embed_texts.return_value = [[1.0, 0.0]]
    collection = get_collection(str(tmp_path))  # nothing added

    graph = build_graph(
        collection,
        top_k=3,
        max_distance=0.65,
        fallback_message="Sorry, no answer found.",
        **_BUILD_GRAPH_DEFAULTS,
    )

    result = graph.invoke({"question": "irrelevant text, embed_texts is mocked"})

    assert result["is_confident"] is False
    assert result["answer"] == "Sorry, no answer found."
    mock_ask_llm.assert_not_called()
