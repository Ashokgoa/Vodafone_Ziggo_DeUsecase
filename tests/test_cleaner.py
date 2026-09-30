"""Tests for app.ingestion.cleaner."""

from app.ingestion.cleaner import clean_text


def test_clean_text_strips_whitespace_and_drops_blank_lines() -> None:
    raw = "\n  Internet  \n\n\n   \n  Snel en betrouwbaar.  \n"

    cleaned = clean_text(raw)

    assert cleaned == "Internet\nSnel en betrouwbaar."
