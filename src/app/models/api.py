"""Request/response shapes for the FastAPI /ask endpoint.

Pydantic models double as documentation and validation: FastAPI rejects
any request that doesn't match AskRequest before our own code ever runs,
and generates the interactive API docs from these same field definitions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    """The customer's question.

    max_length is a simple abuse/cost guard: a customer question doesn't
    need to be an essay, and this also bounds how much text we'd embed and
    (on the confident path) send to the LLM.
    """

    question: str = Field(..., min_length=1, max_length=500)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        # min_length=1 alone would still accept a string of only spaces.
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class AskResponse(BaseModel):
    """The assistant's answer.

    is_confident lets a frontend distinguish a real, grounded answer from
    the fallback message -- e.g. to style it differently or log it for
    review -- without having to string-match the fallback text.
    """

    answer: str
    is_confident: bool
