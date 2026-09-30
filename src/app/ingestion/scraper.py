"""Fetch the raw HTML of a single web page.

This is the first stage of the ingestion pipeline:

    scrape HTML -> extract -> clean -> chunk -> embed -> store

Scraping is intentionally limited to a single page fetched on demand — no
crawling, no following links — matching the assignment's scope and staying
respectful of the target site.
"""

from __future__ import annotations

import requests

# Identifies this program to the server we fetch from, instead of relying on
# the HTTP library's generic default User-Agent. This is not a secret, so it
# is a plain constant rather than something read from configuration.
_USER_AGENT = "ziggo-rag-assistant/0.1 (educational take-home assignment)"

# How long to wait for a response before giving up. A fixed, generous default
# is enough for a page fetched interactively/offline; there's no need to make
# this configurable until a real scenario requires it.
_DEFAULT_TIMEOUT_SECONDS = 10


def fetch_html(url: str, timeout: float = _DEFAULT_TIMEOUT_SECONDS) -> str:
    """Fetch a single page and return its raw HTML as text.

    Raises requests.exceptions.RequestException (e.g. a connection failure)
    or requests.exceptions.HTTPError (e.g. a 404/500 response) if the fetch
    does not succeed. We let these propagate as-is rather than wrapping them
    in a custom exception: this function runs inside an offline ingestion
    step, so a precise, standard exception is more useful than an extra
    abstraction layer.
    """
    response = requests.get(
        url,
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
    )
    # raise_for_status() turns a 404/500/etc. response into an exception
    # instead of silently treating an error page's HTML as real content.
    response.raise_for_status()
    return response.text
