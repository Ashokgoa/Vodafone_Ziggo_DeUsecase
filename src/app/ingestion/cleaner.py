"""Normalize extracted text into clean, readable plain text.

This is the third stage of the ingestion pipeline:

    scrape HTML -> extract -> CLEAN -> chunk -> embed -> store

extract_text() can still leave behind a lot of noise: blank lines and
leading/trailing whitespace on each line. This function's only job is to
tidy that up — it knows nothing about HTML, only about text.
"""

from __future__ import annotations


def clean_text(text: str) -> str:
    """Strip whitespace and drop empty lines, returning tidy plain text."""
    lines = (line.strip() for line in text.splitlines())
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines)
