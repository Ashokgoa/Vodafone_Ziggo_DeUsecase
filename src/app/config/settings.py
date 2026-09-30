"""Application configuration, loaded from environment variables.

Centralizing configuration here means every other module reads settings
from one place, instead of each module calling os.environ.get(...) with
its own scattered default values and hardcoded strings. It also keeps
secrets and environment-specific values (a target URL, an LLM API key, a
vector-store path, ...) out of the source code, as required by the
assignment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load variables from a local .env file (if one exists) into the process
# environment. This only matters for local development: in Docker or CI the
# environment variables are set directly, and load_dotenv() silently does
# nothing if no .env file is present. Its return value (whether a .env file
# was found) is intentionally unused.
_ = load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Typed view over the environment variables the app depends on.

    frozen=True makes instances immutable: once created, a Settings object
    cannot be accidentally modified elsewhere in the code.
    """

    app_env: str
    log_level: str
    target_url: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    vector_store_dir: str
    retrieval_top_k: int
    retrieval_max_distance: float
    fallback_message: str
    llm_model: str
    llm_max_tokens: int


def get_settings() -> Settings:
    """Read the current environment variables and return a Settings instance.

    This re-reads the environment on every call rather than caching a
    single instance. For the handful of values we have, the cost is
    negligible, and it keeps tests simple: a test can set an environment
    variable and immediately see get_settings() reflect it.
    """
    return Settings(
        app_env=os.getenv("APP_ENV", "local"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        target_url=os.getenv("ZIGGO_URL", "https://www.ziggo.nl/internet"),
        # int(...) deliberately raises a clear error if someone puts a
        # non-numeric value in the environment, rather than silently
        # falling back to a default that hides the mistake.
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        # A small multilingual model: our page content is in Dutch, so an
        # English-only model would silently produce worse embeddings.
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
        ),
        # Relative to the working directory. Already covered by .gitignore
        # (the "data/" entry), so vector-store files never get committed.
        vector_store_dir=os.getenv("VECTOR_STORE_DIR", "data/vectorstore"),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "3")),
        # Cosine distance threshold below which a match is trusted enough
        # to answer from. Chosen empirically: on our real page, relevant
        # questions scored 0.24-0.61 and unrelated ones scored 0.74-0.97,
        # so 0.65 sits in the gap between them (see README for the
        # measurement). Above this, we treat retrieval as "no answer
        # found" rather than risk answering from an irrelevant chunk.
        retrieval_max_distance=float(os.getenv("RETRIEVAL_MAX_DISTANCE", "0.65")),
        # Shown to the customer whenever retrieval isn't confident, instead
        # of letting the LLM guess. Configurable so wording can change (or
        # be translated) without a code change.
        fallback_message=os.getenv(
            "FALLBACK_MESSAGE",
            "I couldn't find a confident answer to that question based on "
            "the information on this page. Please contact a Ziggo advisor "
            "for further help.",
        ),
        # Haiku is deliberately the default: our context per question is
        # small (a few short chunks), so a larger, pricier model wouldn't
        # meaningfully improve answer quality here.
        llm_model=os.getenv("LLM_MODEL", "claude-haiku-4-5"),
        # A short cap: customer-facing answers here should be a few
        # sentences, not an essay, and this also bounds cost per question.
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "500")),
    )
