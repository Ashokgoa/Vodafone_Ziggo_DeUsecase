"""Tests for app.ingestion.scraper.

These mock the HTTP call (via unittest.mock, from the standard library) so
the test suite never touches the real network — it stays fast, free, and
unaffected by whether the Ziggo site is reachable or has changed.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.ingestion.scraper import fetch_html


@patch("app.ingestion.scraper.requests.get")
def test_fetch_html_returns_page_text(mock_get: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.text = "<html>hello</html>"
    mock_get.return_value = mock_response

    html = fetch_html("https://example.com/page")

    assert html == "<html>hello</html>"
    mock_response.raise_for_status.assert_called_once()

    # Confirm the request identified itself and won't hang forever, without
    # depending on the exact argument order used to call requests.get.
    _, kwargs = mock_get.call_args
    assert "User-Agent" in kwargs["headers"]
    assert kwargs["timeout"] > 0


@patch("app.ingestion.scraper.requests.get")
def test_fetch_html_raises_on_http_error(mock_get: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
    mock_get.return_value = mock_response

    with pytest.raises(requests.HTTPError):
        fetch_html("https://example.com/missing")
