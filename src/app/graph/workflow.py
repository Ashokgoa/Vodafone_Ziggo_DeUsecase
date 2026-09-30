"""The LangGraph workflow: embed -> retrieve -> confidence branch -> answer.

This is the full assignment-required flow, end to end:

    ... retrieve relevant chunks -> LANGGRAPH WORKFLOW -> LLM -> answer

A LangGraph graph is a small state machine: each node is a plain function
that reads a shared state dict and returns the fields it wants to update,
and edges say which node runs next -- including conditional edges, where a
function inspects the state and picks the next node. We use it here so the
"did we find something worth answering from?" branch is an explicit,
visible part of the graph's structure, not an if-statement buried inside a
larger function -- matching the assignment's "agentic backend" requirement
and making the workflow easy to diagram and to extend with more steps
later.

generate_answer calls the LLM (see llm_client.py) only on the confident
path -- fallback_answer returns a fixed message instead, so the LLM is
never asked to answer without real grounding, and low-confidence questions
cost nothing.
"""

from __future__ import annotations

from typing import TypedDict

from chromadb.api.models.Collection import Collection
from langgraph.graph import END, StateGraph

from app.graph.llm_client import ask_llm
from app.retrieval.embeddings import embed_texts
from app.retrieval.retrieve import retrieve


class GraphState(TypedDict, total=False):
    """Data that flows through the graph.

    total=False means no field is required upfront: only "question" is set
    before the graph starts running, and each node fills in more fields as
    it completes (the standard LangGraph pattern for a state that
    accumulates as it flows through the graph).
    """

    question: str
    question_embedding: list[float]
    chunks: list[dict]
    is_confident: bool
    answer: str


def build_graph(
    collection: Collection,
    embedding_model: str,
    top_k: int,
    max_distance: float,
    fallback_message: str,
    llm_model: str,
    llm_max_tokens: int,
):
    """Construct and compile the embed -> retrieve -> answer/fallback graph.

    The vector store collection, model name, and thresholds are captured
    here as closures rather than carried in the state: they're fixed
    configuration for the whole app, not per-question data.
    """

    def embed_question(state: GraphState) -> dict:
        [vector] = embed_texts([state["question"]], embedding_model)
        return {"question_embedding": vector}

    def retrieve_chunks(state: GraphState) -> dict:
        result = retrieve(state["question_embedding"], collection, top_k, max_distance)
        return {"chunks": result.chunks, "is_confident": result.is_confident}

    def generate_answer(state: GraphState) -> dict:
        context = "\n\n".join(chunk["text"] for chunk in state["chunks"])
        answer = ask_llm(state["question"], context, llm_model, llm_max_tokens)
        return {"answer": answer}

    def fallback_answer(state: GraphState) -> dict:
        return {"answer": fallback_message}

    def route_on_confidence(state: GraphState) -> str:
        return "generate_answer" if state["is_confident"] else "fallback_answer"

    graph = StateGraph(GraphState)
    graph.add_node("embed_question", embed_question)
    graph.add_node("retrieve_chunks", retrieve_chunks)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("fallback_answer", fallback_answer)

    graph.set_entry_point("embed_question")
    graph.add_edge("embed_question", "retrieve_chunks")
    graph.add_conditional_edges(
        "retrieve_chunks",
        route_on_confidence,
        {"generate_answer": "generate_answer", "fallback_answer": "fallback_answer"},
    )
    graph.add_edge("generate_answer", END)
    graph.add_edge("fallback_answer", END)

    return graph.compile()
