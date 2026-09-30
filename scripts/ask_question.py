"""Manual developer helper: ask a question through the full LangGraph workflow.

Not part of the automated test suite. Requires ingestion to have already
run at least once (see scripts/inspect_ingestion.py) so the vector store
has chunks to retrieve from.

Run with:
    python scripts/ask_question.py "Welke internetsnelheden biedt Ziggo aan?"
"""

import sys

from app.config.settings import get_settings
from app.graph.workflow import build_graph
from app.retrieval.vector_store import get_collection


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/ask_question.py "your question"')
        raise SystemExit(1)
    question = sys.argv[1]

    settings = get_settings()
    collection = get_collection(settings.vector_store_dir)
    graph = build_graph(
        collection,
        settings.embedding_model,
        settings.retrieval_top_k,
        settings.retrieval_max_distance,
        settings.fallback_message,
        settings.llm_model,
        settings.llm_max_tokens,
    )

    result = graph.invoke({"question": question})
    print(f"Question: {question}")
    print(f"is_confident: {result['is_confident']}")
    print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    main()
