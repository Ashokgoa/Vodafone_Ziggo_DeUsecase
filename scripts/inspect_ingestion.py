"""Manual developer helper: run scrape -> extract -> clean and print each stage.

Not part of the automated test suite or the API. The core pipeline
intentionally keeps intermediate results in memory only (see README); this
script exists purely so a developer can eyeball what each stage produces.

Run with:
    python scripts/inspect_ingestion.py
"""

from app.config.settings import get_settings
from app.ingestion.chunker import chunk_text
from app.ingestion.cleaner import clean_text
from app.ingestion.extractor import extract_text
from app.ingestion.scraper import fetch_html
from app.retrieval.embeddings import embed_texts
from app.retrieval.retrieve import retrieve
from app.retrieval.vector_store import add_chunks, get_collection


def main() -> None:
    settings = get_settings()
    print(f"Fetching: {settings.target_url}\n")

    html = fetch_html(settings.target_url)
    print(f"Raw HTML length: {len(html)} characters")

    extracted = extract_text(html)
    print(f"Extracted text length: {len(extracted)} characters")

    cleaned = clean_text(extracted)
    print(f"Cleaned text length: {len(cleaned)} characters")

    chunks = chunk_text(cleaned, settings.chunk_size, settings.chunk_overlap)
    print(
        f"Chunks: {len(chunks)} "
        f"(chunk_size={settings.chunk_size}, chunk_overlap={settings.chunk_overlap})\n"
    )

    print("--- Chunks ---")
    for i, chunk in enumerate(chunks, 1):
        print(f"[Chunk {i}/{len(chunks)}, {len(chunk)} chars]")
        print(chunk)
        print()

    print(f"Embedding {len(chunks)} chunks with model: {settings.embedding_model}")
    print("(first run downloads the model — this can take a while)\n")
    vectors = embed_texts(chunks, settings.embedding_model)
    print(f"Got {len(vectors)} vectors, each with {len(vectors[0])} dimensions")
    print(f"First vector, first 8 numbers: {vectors[0][:8]}\n")

    print(f"Storing chunks in vector store: {settings.vector_store_dir}")
    collection = get_collection(settings.vector_store_dir)
    add_chunks(collection, chunks, vectors, source_url=settings.target_url)
    print(f"Stored. Collection now has {collection.count()} chunks.\n")

    # Two example questions: one genuinely answerable from this page, one
    # not, to demonstrate retrieve()'s low-confidence handling for real
    # rather than just in unit tests.
    example_questions = [
        "Welke internetsnelheden biedt Ziggo aan?",
        "Wat is de hoofdstad van Frankrijk?",
    ]
    for question in example_questions:
        print(f"\n=== Question: {question!r} ===")
        [question_vector] = embed_texts([question], settings.embedding_model)
        result = retrieve(
            question_vector,
            collection,
            top_k=settings.retrieval_top_k,
            max_distance=settings.retrieval_max_distance,
        )
        print(f"is_confident: {result.is_confident}")
        if result.is_confident:
            for rank, match in enumerate(result.chunks, 1):
                print(f"\n#{rank} (distance={match['distance']:.4f}):")
                print(match["text"])
        else:
            print("No confident match -> this is where a safe fallback answer kicks in.")


if __name__ == "__main__":
    main()
