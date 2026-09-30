"""Split cleaned text into smaller, overlapping chunks.

This is the fourth stage of the ingestion pipeline:

    scrape HTML -> extract -> clean -> CHUNK -> embed -> store

Chunking exists because both embeddings and the LLM work best over small,
focused pieces of text: a single embedding for an entire page blurs
together unrelated topics (e.g. pricing and FAQs), and retrieval needs to
return only the specific piece of the page relevant to a question.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks.

    Uses RecursiveCharacterTextSplitter, which tries to cut on a paragraph
    break first, then a line break, then a sentence, then a word boundary,
    and only cuts mid-word as a last resort. chunk_overlap repeats a small
    slice of text at the start of the next chunk, so a chunk near a
    boundary still carries a hint of what came right before it.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)
