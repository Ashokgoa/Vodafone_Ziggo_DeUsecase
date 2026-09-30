"""Tests for app.ingestion.chunker."""

from app.ingestion.chunker import chunk_text

_SAMPLE_TEXT = (
    "Ziggo biedt internet met verschillende snelheden. "
    "Het 200 Mbit/s pakket is geschikt voor lichte gebruikers. "
    "Het 500 Mbit/s pakket is geschikt voor gezinnen. "
    "Het 1 Gbit/s pakket is geschikt voor zware gebruikers. "
    "Het 2 Gbit/s pakket is het snelste dat Ziggo aanbiedt."
)


def test_chunk_text_splits_long_text_into_multiple_chunks() -> None:
    chunks = chunk_text(_SAMPLE_TEXT, chunk_size=80, chunk_overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk  # no empty chunks
        assert len(chunk) <= 80


def test_chunk_text_keeps_short_text_as_a_single_chunk() -> None:
    text = "Kort stukje tekst."

    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)

    # Text that already fits within chunk_size shouldn't be split at all.
    assert chunks == [text]


def test_chunk_text_overlaps_consecutive_chunks() -> None:
    chunks = chunk_text(_SAMPLE_TEXT, chunk_size=80, chunk_overlap=20)

    # Consecutive chunks should share at least one word, proving the
    # overlap setting actually repeats text at each boundary rather than
    # cutting the text into disjoint pieces.
    for first, second in zip(chunks, chunks[1:]):
        shared_words = set(first.split()) & set(second.split())
        assert shared_words, f"expected shared words between {first!r} and {second!r}"
