"""FastAPI app exposing the assistant as POST /ask.

This is the final stage of the assignment's required flow:

    ... LangGraph workflow -> LLM -> customer-facing answer -> FASTAPI /ask

The compiled LangGraph graph is built once, in lifespan(), when the server
starts, and stored on app.state -- not rebuilt per request. Building it
opens the vector store and wires up the workflow's nodes, none of which
depends on the specific question being asked, so redoing it every request
would be pure waste.

Run locally with:
    uvicorn app.api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from app.config.settings import get_settings
from app.graph.workflow import build_graph
from app.ingestion.pipeline import run_ingestion
from app.models.api import AskRequest, AskResponse
from app.retrieval.vector_store import get_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    collection = get_collection(settings.vector_store_dir)
    if collection.count() == 0:
        # First startup (or a fresh volume with nothing in it yet):
        # populate the vector store automatically, so `docker compose up`
        # alone is enough to get a working API -- no separate manual
        # ingestion step required. On a later restart with a persisted
        # volume, the store already has chunks and this is skipped.
        run_ingestion(settings)
    app.state.graph = build_graph(
        collection,
        settings.embedding_model,
        settings.retrieval_top_k,
        settings.retrieval_max_distance,
        settings.fallback_message,
        settings.llm_model,
        settings.llm_max_tokens,
    )
    yield
    # Nothing to clean up: Chroma's PersistentClient needs no explicit close.


app = FastAPI(title="Ziggo RAG Assistant", lifespan=lifespan)


def get_graph(request: Request):
    """Dependency returning the compiled graph built at startup.

    Reading it via a dependency (rather than the endpoint reaching into
    app.state directly) means tests can swap in a fake graph with
    app.dependency_overrides, without a real vector store, embedding
    model, or LLM -- see tests/test_api.py.
    """
    return request.app.state.graph


@app.get("/health")
def health() -> dict:
    """Basic liveness check, used by Docker Compose's healthcheck (Step 12)."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, graph=Depends(get_graph)) -> AskResponse:
    try:
        result = graph.invoke({"question": request.question})
    except Exception as exc:
        # Deliberately broad: a failure here could come from embedding, the
        # vector store, or the LLM call, and in every case the customer
        # should see a safe, generic message -- never a raw stack trace,
        # and never a crashed process.
        raise HTTPException(
            status_code=502,
            detail="Something went wrong while answering your question. Please try again.",
        ) from exc

    return AskResponse(answer=result["answer"], is_confident=result["is_confident"])
