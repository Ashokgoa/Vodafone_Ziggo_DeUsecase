"""Tests for the FastAPI /ask endpoint.

Overrides the get_graph dependency with a fake graph via
app.dependency_overrides -- the officially documented FastAPI pattern for
testing endpoints without their real dependencies. The TestClient is also
deliberately NOT used as a context manager, so the app's lifespan (which
would build a real graph: a real vector store, a real embedding model,
a real LLM client) never runs.
"""

from fastapi.testclient import TestClient

from app.api.main import app, get_graph


class _FakeGraph:
    """Stands in for the compiled LangGraph graph in tests."""

    def __init__(self, result: dict) -> None:
        self._result = result

    def invoke(self, state: dict) -> dict:
        return self._result


class _BrokenGraph:
    """Simulates a failure inside the graph (e.g. the LLM API is down)."""

    def invoke(self, state: dict) -> dict:
        raise RuntimeError("simulated LLM outage")


def test_ask_returns_the_answer_and_confidence() -> None:
    app.dependency_overrides[get_graph] = lambda: _FakeGraph(
        {"answer": "Ziggo biedt tot 2 Gbit/s.", "is_confident": True}
    )
    client = TestClient(app)

    response = client.post("/ask", json={"question": "Wat is de snelste optie?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Ziggo biedt tot 2 Gbit/s.", "is_confident": True}

    app.dependency_overrides.clear()


def test_ask_rejects_a_blank_question() -> None:
    app.dependency_overrides[get_graph] = lambda: _FakeGraph(
        {"answer": "should never be reached", "is_confident": True}
    )
    client = TestClient(app)

    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422  # FastAPI's validation-error status

    app.dependency_overrides.clear()


def test_ask_rejects_a_question_over_the_length_limit() -> None:
    app.dependency_overrides[get_graph] = lambda: _FakeGraph(
        {"answer": "should never be reached", "is_confident": True}
    )
    client = TestClient(app)

    response = client.post("/ask", json={"question": "a" * 501})

    assert response.status_code == 422

    app.dependency_overrides.clear()


def test_ask_returns_a_safe_error_when_the_graph_fails() -> None:
    app.dependency_overrides[get_graph] = lambda: _BrokenGraph()
    client = TestClient(app)

    response = client.post("/ask", json={"question": "Welke snelheden zijn er?"})

    assert response.status_code == 502
    assert "went wrong" in response.json()["detail"].lower()
    # The real exception message must never reach the customer.
    assert "simulated LLM outage" not in response.text

    app.dependency_overrides.clear()


def test_health_check_does_not_need_the_graph() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
