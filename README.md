# Ziggo RAG Assistant

## Objective

Take-home assignment (Big Data Engineer): a simple customer-facing AI assistant that
answers questions about a Ziggo web page (`https://www.ziggo.nl/internet` by default)
using Retrieval-Augmented Generation, with an agentic backend built on LangGraph,
exposed through a FastAPI `POST /ask` endpoint, runnable end-to-end via Docker
Compose.

## Current status

**Complete.** The full required pipeline works end-to-end and is covered by 30
automated tests:

```
scrape → extract → clean → chunk → embed → store (local vector store)
  → retrieve (with low-confidence handling) → LangGraph workflow → LLM
  → FastAPI /ask
```

`Dockerfile` and `docker-compose.yml` exist and every dependency has been verified
to have a pre-built Linux wheel (see `docs/ARCHITECTURE.md`), but **the actual
`docker compose up` has not yet been run and confirmed** in this development
environment (Docker isn't installed here) — see [Limitations](#limitations-and-next-steps).

## Architecture

Full diagrams (local + an AWS representation, with an explicit "required vs.
production-scale" breakdown) are in **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**.

In short, locally:

```
Ingestion (runs once, only if the vector store is empty):
  Ziggo page --scrape--> HTML --extract--> text --clean--> text --chunk--> chunks
    --embed (local model)--> vectors --store--> Chroma (Docker volume)

Query (per customer question):
  Customer --POST /ask--> FastAPI --> LangGraph workflow
    --embed question--> search Chroma --> confident?
      yes --> Claude Haiku 4.5 --> answer
      no  --> fixed fallback message (no LLM call)
    --> FastAPI --> Customer
```

## Technology choices and why

| Concern | Library | Why |
|---|---|---|
| Config from env vars | `python-dotenv` | Loads a local `.env` for development only; in Docker/CI, env vars are set directly |
| HTTP requests | `requests` | Simple, synchronous; scraping runs once, so async brings no benefit |
| HTML parsing | `beautifulsoup4` | Standard, readable API for stripping non-content tags |
| Text chunking | `langchain-text-splitters` | `RecursiveCharacterTextSplitter` prefers paragraph/sentence/word boundaries over cutting at a fixed character count |
| Embeddings | `sentence-transformers` | Local, free, multilingual model (our content is Dutch) — no API key needed for this stage, keeps ingestion fully offline |
| Vector store | `chromadb` | Embedded (no server), persists to a folder — satisfies the "local vector store" requirement; stores text + vector + metadata together |
| Agentic workflow | `langgraph` | Required by the assignment; models the confidence branch as an explicit graph node rather than a hidden `if` |
| LLM | `anthropic` (Claude Haiku 4.5) | Called via the plain SDK, not a LangChain model wrapper — one fewer dependency. Haiku: our per-question context is small, so a larger model wouldn't improve quality here |
| API | `fastapi` + `uvicorn` | Automatic request validation, interactive docs, the framework the assignment names explicitly |

## Project layout

```
src/app/
  config/      settings loaded from environment variables
  models/      request/response schemas (FastAPI)
  ingestion/   scrape, extract, clean, chunk, and the end-to-end pipeline
  retrieval/   embeddings, vector store, retrieval with confidence handling
  graph/       LangGraph workflow + the LLM client
  api/         FastAPI app (POST /ask, GET /health)
tests/         pytest tests (one file per module, all fast — mocked network/model/LLM calls)
scripts/       manual developer helpers (not used by the API or Docker image)
docs/          architecture diagrams
Dockerfile
docker-compose.yml
```

## Local setup (without Docker)

Requires **Python 3.11.x**.

```powershell
# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate

# Install the project in editable mode with dev dependencies
pip install -e ".[dev]"

# Create your local config
copy .env.example .env                # macOS/Linux: cp .env.example .env
```

Then edit `.env` and set a real `ANTHROPIC_API_KEY` (see below) — everything else has
a sensible default.

## Configuration

All settings are environment variables, read once in `app.config.settings`.

| Variable | Default | Meaning |
|---|---|---|
| `APP_ENV` | `local` | Which environment the app is running in |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `ZIGGO_URL` | `https://www.ziggo.nl/internet` | The page the assistant answers questions about |
| `CHUNK_SIZE` | `500` | Target characters per chunk |
| `CHUNK_OVERLAP` | `50` | Characters shared between consecutive chunks |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Local sentence-transformers model (multilingual, for Dutch content) |
| `VECTOR_STORE_DIR` | `data/vectorstore` | Where Chroma persists its files |
| `RETRIEVAL_TOP_K` | `3` | How many chunks to retrieve per question |
| `RETRIEVAL_MAX_DISTANCE` | `0.65` | Cosine-distance threshold for "confident enough to answer" — chosen empirically, see inline comment in `settings.py` |
| `FALLBACK_MESSAGE` | *(see `.env.example`)* | Shown when retrieval isn't confident |
| `ANTHROPIC_API_KEY` | *(none — required)* | Your key from console.anthropic.com. Never commit a real value |
| `LLM_MODEL` | `claude-haiku-4-5` | The model that writes the final answer |
| `LLM_MAX_TOKENS` | `500` | Cap on the answer's length |

### Getting an `ANTHROPIC_API_KEY`

1. Sign up / log in at **console.anthropic.com**.
2. Add a small amount of billing credit (API usage is pay-as-you-go). Testing this
   project costs fractions of a cent — Haiku 4.5 is $1/$5 per million tokens, and each
   question uses a few hundred tokens at most.
3. **API Keys** → **Create Key** → copy the value (starts with `sk-ant-`) into your
   `.env` as `ANTHROPIC_API_KEY=...`.

## Running ingestion

You don't need to run this manually — the API does it automatically on first startup
if the vector store is empty (see `app/api/main.py`'s `lifespan`). To inspect each
pipeline stage by hand instead (useful for debugging or just seeing what the scraper
actually extracts):

```powershell
python scripts/inspect_ingestion.py
```

This prints the raw HTML length, extracted/cleaned text, every chunk, the embedding
vectors' shape, and two example questions run through retrieval (one answerable, one
not) — all against the real, live Ziggo page.

## Running the API locally (without Docker)

```powershell
uvicorn app.api.main:app --reload
```

First startup ingests automatically (downloads the embedding model, scrapes, embeds —
can take a couple of minutes). Then:

```powershell
curl http://localhost:8000/health
# {"status":"ok"}

curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" `
  -d '{"question": "Welke internetsnelheden biedt Ziggo aan?"}'
# {"answer": "Ziggo biedt internetsnelheden van 200 Mbit/s tot 2 Gbit/s...", "is_confident": true}

curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" `
  -d '{"question": "Wat is de hoofdstad van Frankrijk?"}'
# {"answer": "I couldn't find a confident answer to that question based on the
#   information on this page. Please contact a Ziggo advisor for further help.",
#   "is_confident": false}
```

Interactive API docs (Swagger UI, auto-generated by FastAPI): **http://localhost:8000/docs**

## Running via Docker Compose

```powershell
copy .env.example .env      # then fill in a real ANTHROPIC_API_KEY
docker compose up --build
```

First startup takes a few minutes (embedding model download + scrape + embed, same
as running locally). The vector store persists in a Docker volume, so subsequent
`docker compose restart` calls skip re-ingestion and start in seconds. Tear down with
`docker compose down` (keeps the volume) or `docker compose down -v` (also deletes it).

> **Not yet verified end-to-end** — see [Limitations](#limitations-and-next-steps).

## Running tests

```powershell
pytest
```

30 tests, all fast (a few seconds), and none touch the real network, a real embedding
model download, or a real (billed) LLM call — everything external is mocked. A
handful of tests use a real, temporary Chroma collection (fast and local, so no
mocking needed there).

## Limitations and next steps

**Known limitation, not a bug:** the Ziggo page renders pricing information via
client-side JavaScript, which a static HTML scrape can't see — our scraped content
has no price text at all. Asking about pricing correctly triggers the low-confidence
fallback rather than inventing a number.

**Pending verification:** `docker compose up` has not been run in this development
environment (Docker isn't installed here). The Dockerfile's dependency resolution has
been verified for Linux (see `docs/ARCHITECTURE.md`), but the actual build/run/curl
cycle still needs to be confirmed on a machine with Docker.

**Reasonable next improvements** (deliberately not built, to avoid over-engineering a
take-home assignment):
- Scheduled re-ingestion (the page's content could change; we only re-ingest when the
  store is empty)
- A managed vector store for multi-page/high-concurrency scale (see `docs/ARCHITECTURE.md`)
- Structured logging and request tracing
- Rate limiting on `/ask`
- An automated evaluation set for answer quality, beyond the two example questions
  used during development
- Language detection, so the fallback message matches the question's language
  instead of always being in English
