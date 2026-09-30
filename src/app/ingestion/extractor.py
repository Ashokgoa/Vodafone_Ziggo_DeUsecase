"""Extract the human-readable text from a page's raw HTML.

This is the second stage of the ingestion pipeline:

    scrape HTML -> EXTRACT -> clean -> chunk -> embed -> store

Extraction is deliberately kept separate from cleaning (cleaner.py): this
function's only job is to throw away markup and non-content elements
(scripts, navigation, footers, ...) and return whatever visible text is
left. Normalizing whitespace happens in the next stage.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

# Tags that never contain content a customer would want answered from,
# regardless of which page we scrape — always safe to discard.
_TAGS_TO_REMOVE = ["script", "style", "noscript", "nav", "header", "footer", "svg"]


def extract_text(html: str) -> str:
    """Parse HTML and return its visible text, with boilerplate tags removed."""
    soup = BeautifulSoup(html, "html.parser")

    for tag_name in _TAGS_TO_REMOVE:
        # decompose() removes the tag AND everything nested inside it, so a
        # <nav><a>...</a></nav> disappears entirely rather than leaving the
        # link text behind.
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Some frameworks (e.g. Vue) inject structured data meant for search
    # engines, not customers, via a non-<script> element such as
    # <component :is="'script'" type="application/ld+json">. Matching by
    # tag name alone misses this, so we also remove anything carrying this
    # specific type attribute, whatever tag it's wrapped in.
    for tag in soup.find_all(attrs={"type": "application/ld+json"}):
        tag.decompose()

    # separator="\n" keeps each block-level element on its own line instead
    # of gluing adjacent text together (e.g. a heading run into the next
    # paragraph); cleaner.py normalizes the resulting whitespace.
    return soup.get_text(separator="\n")
