"""The end-to-end ingestion pipeline: scrape -> extract -> clean -> chunk ->
embed -> store, as one reusable function.

Each stage already has its own module and its own tests (scraper.py,
extractor.py, cleaner.py, chunker.py, embeddings.py, vector_store.py); this
module just wires them together in order. It exists so the API (which
needs to run this automatically on first startup -- see app/api/main.py)
and scripts/inspect_ingestion.py (which prints each stage for a developer
to inspect) call the exact same sequence, instead of two copies that could
quietly drift apart over time.
"""

from __future__ import annotations

from app.config.settings import Settings
from app.ingestion.chunker import chunk_text
from app.ingestion.cleaner import clean_text
from app.ingestion.extractor import extract_text
from app.ingestion.scraper import fetch_html
from app.retrieval.embeddings import embed_texts
from app.retrieval.vector_store import add_chunks, get_collection


def run_ingestion(settings: Settings) -> int:
    """Scrape, clean, chunk, embed, and store the configured page.

    Returns the number of chunks stored, so a caller can log/print it
    without a separate query against the collection.
    """
    html = fetch_html(settings.target_url)
    extracted = extract_text(html)
    cleaned = clean_text(extracted)
    chunks = chunk_text(cleaned, settings.chunk_size, settings.chunk_overlap)
    vectors = embed_texts(chunks, settings.embedding_model)

    collection = get_collection(settings.vector_store_dir)
    add_chunks(collection, chunks, vectors, source_url=settings.target_url)
    return len(chunks)
