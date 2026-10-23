from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from app import config
from app.graph import build_graph
from app.models import get_embedder, get_llm
from app.store import VectorStore

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Collection is per provider: vectors from different embedding models are not comparable.
    store = VectorStore(get_embedder(), name=f"docs-{config.LLM_PROVIDER}")
    if store.col.count() == 0:  # Cloud Run disk is ephemeral: index bundled docs on cold start
        store.ingest_dir(config.DOCS_DIR)
    state["graph"] = build_graph(get_llm(), store)
    state["store"] = store
    yield


app = FastAPI(title="LangGraph Agentic RAG", lifespan=lifespan)


class Ask(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "provider": config.LLM_PROVIDER,
            "chunks": state["store"].col.count()}


@app.post("/ask")
def ask(body: Ask):
    out = state["graph"].invoke({"question": body.question})
    return {"answer": out["answer"], "grounded": out["grounded"],
            "sources": sorted({d["source"] for d in out.get("documents", [])}),
            "retries": out.get("retries", 0), "trace": out["trace"]}
