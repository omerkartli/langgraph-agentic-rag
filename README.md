# LangGraph Agentic RAG

A self-correcting RAG service built with **LangGraph**, **Chroma** and **FastAPI**, designed to run on **Google Cloud** (Cloud Run + Vertex AI Gemini and embeddings).

It follows up my MSc work on RAG evaluation ([ragopt](https://github.com/omerkartli/ragopt)): there the pipeline was a fixed retrieve-then-generate script; here retrieval quality and answer faithfulness are checked at runtime and the agent reacts to them.

## How the agent works

```
route ─┬─ direct ──────────────────────────► generate_direct ─► END
       └─ retrieve ─► grade ─┬─ relevant ─► generate ─► check ─┬─ grounded ─► END
             ▲               │                                  └─ not grounded ─┐
             │               └─ none relevant ─┐                                  │
             └──────────── rewrite ◄───────────┴──────── (while retries remain) ◄─┘
                               └─ retries exhausted ─► fallback ("could not find a reliable answer")
```

- **route**: small talk skips retrieval.
- **grade**: an LLM judge drops irrelevant chunks before generation.
- **rewrite**: if nothing relevant is found (or the answer is not grounded), the query is rewritten and retrieval is retried, up to `MAX_RETRIES`.
- **check**: a groundedness check on the final answer. If retries run out, the response is returned with `grounded: false` rather than hiding the problem.
- Every response includes the `trace` of visited nodes, the `sources` used and the retry count.

## Stack

| Concern | Choice |
|---|---|
| Orchestration | LangGraph `StateGraph` with conditional edges |
| Vector database | Chroma (HNSW, cosine), paragraph-aware chunking with overlap |
| LLM / embeddings | Vertex AI Gemini + `text-embedding-005` (`LLM_PROVIDER=vertex`) |
| API | FastAPI (`POST /ask`, `GET /health`) |
| Packaging / hosting | Docker, Cloud Run, service-account auth (no API keys) |

## Run locally (offline mode, no credentials)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest                                   # graph logic, retrieval, API
uvicorn app.api:app --reload             # LLM_PROVIDER=fake by default
curl -s localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question":"What is Cloud Run?"}'
```

`fake` mode uses a hashing embedder and a scripted LLM so the control flow can be tested deterministically. It is **not** a quality demo; answers are only as good as lexical overlap.

## Run with Vertex AI

```bash
gcloud auth application-default login
export LLM_PROVIDER=vertex GOOGLE_CLOUD_PROJECT=<project> GOOGLE_CLOUD_LOCATION=europe-west4
uvicorn app.api:app
```

Put your own `.md`/`.txt` files in `data/docs` (or set `DOCS_DIR`); they are indexed on startup when the collection is empty. Collections are named per provider because vectors from different embedding models are not comparable.

## Deploy to Cloud Run

```bash
PROJECT_ID=<project> ./deploy/deploy.sh europe-west4
```

The script enables the required APIs, creates a runtime service account that only has `roles/aiplatform.user`, and deploys from source. Cloud Run disk is ephemeral, so the bundled docs are re-embedded on cold start; for a larger corpus, move the index to a managed store (e.g. Vertex AI Vector Search).

## Status

- Graph logic, retrieval and the API are covered by tests that run offline.
- The Vertex AI provider and the Cloud Run deploy script are written against the documented APIs but have **not yet been run against a live GCP project**; check the first deployment and open an issue if something differs.

## Layout

```
app/graph.py     LangGraph state machine
app/models.py    LLM/embedding interfaces: Vertex AI + offline fake
app/store.py     Chroma store, chunking, ingestion
app/api.py       FastAPI service
deploy/          Cloud Run deployment script
tests/           pytest suite
```
