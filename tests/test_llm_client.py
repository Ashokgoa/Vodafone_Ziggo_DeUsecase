"""Tests for app.graph.llm_client.

Mocks the Anthropic client entirely: these tests must never make a real,
billed API call, and must run without a real API key present.
"""

from unittest.mock import MagicMock, patch

from app.graph.llm_client import ask_llm


@patch("app.graph.llm_client.anthropic.Anthropic")
def test_ask_llm_returns_the_response_text(mock_anthropic_class: MagicMock) -> None:
    mock_client = MagicMock()
    mock_text_block = MagicMock()
    mock_text_block.text = "Ziggo biedt 200 Mbit/s, 500 Mbit/s, 1 Gbit/s en 2 Gbit/s."
    mock_client.messages.create.return_value.content = [mock_text_block]
    mock_anthropic_class.return_value = mock_client

    answer = ask_llm(
        question="Welke snelheden biedt Ziggo aan?",
        context="Internet 200 Mbit/s ... Internet 2 Gbit/s ...",
        model="fake-model",
        max_tokens=100,
    )

    assert answer == "Ziggo biedt 200 Mbit/s, 500 Mbit/s, 1 Gbit/s en 2 Gbit/s."

    # Confirm both the question and the context actually made it into the
    # request, and that our model/max_tokens settings were passed through.
    _, kwargs = mock_client.messages.create.call_args
    sent_content = kwargs["messages"][0]["content"]
    assert "Welke snelheden biedt Ziggo aan?" in sent_content
    assert "Internet 200 Mbit/s" in sent_content
    assert kwargs["model"] == "fake-model"
    assert kwargs["max_tokens"] == 100
