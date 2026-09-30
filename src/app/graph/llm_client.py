"""Generate a customer-facing answer from an LLM, grounded in retrieved chunks.

This is the "LLM" stage of the assignment's required flow:

    ... LangGraph workflow -> LLM -> customer-facing answer

Called from the graph's generate_answer node (see workflow.py), which only
runs once retrieval is confident there's something relevant to answer from
-- the LLM is never asked to answer without real grounding.

We call the plain Anthropic SDK directly rather than a LangChain wrapper
(e.g. langchain-anthropic): LangGraph itself doesn't require using
LangChain's model classes, and calling the SDK directly is one fewer
dependency and stays closer to the officially documented API.
"""

from __future__ import annotations

import anthropic

# Kept out of Settings deliberately: the Anthropic SDK already reads
# ANTHROPIC_API_KEY from the environment on its own (see
# anthropic.Anthropic()'s docs), so there is no reason for this secret to
# also live inside our own Settings object -- one fewer place it could leak
# from (e.g. an accidental log of a Settings instance).
_SYSTEM_PROMPT = (
    "You are a helpful customer support assistant for Ziggo, a Dutch "
    "internet/TV provider. Answer the customer's question using ONLY the "
    "context below, which was extracted from Ziggo's own website. If the "
    "context does not fully answer the question, say so honestly instead "
    "of guessing or inventing details. Reply in the same language the "
    "question was asked in. Keep the answer concise and friendly."
)


def ask_llm(question: str, context: str, model: str, max_tokens: int) -> str:
    """Ask the LLM to answer the question, grounded only in the given context."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    # A plain text answer with no tool use is always exactly one text block.
    return response.content[0].text
