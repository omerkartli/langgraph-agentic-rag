"""Chroma vector store (HNSW, cosine) with chunking and ingestion."""
import hashlib
from pathlib import Path

import chromadb

from app import config
from app.models import Embedder


def chunk(text: str, size: int = 600, overlap: int = 100) -> list[str]:
    """Paragraph-aware chunking with character overlap."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > size:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n" + p
        else:
            cur = f"{cur}\n\n{p}".strip()
    if cur:
        chunks.append(cur)
    return chunks


class VectorStore:
    def __init__(self, embedder: Embedder, path: str | None = None, name: str = "docs"):
        client = (chromadb.PersistentClient(path=path or config.CHROMA_PATH)
                  if path != ":memory:" else chromadb.EphemeralClient())
        self.col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        self.embedder = embedder

    def add_text(self, text: str, source: str) -> int:
        parts = chunk(text)
        if not parts:
            return 0
        ids = [hashlib.md5(f"{source}:{i}:{c}".encode()).hexdigest() for i, c in enumerate(parts)]
        self.col.upsert(ids=ids, documents=parts,
                        embeddings=self.embedder.embed(parts),
                        metadatas=[{"source": source, "chunk": i} for i in range(len(parts))])
        return len(parts)

    def ingest_dir(self, directory: str) -> int:
        n = 0
        for f in sorted(Path(directory).glob("**/*")):
            if f.suffix.lower() in {".txt", ".md"}:
                n += self.add_text(f.read_text(encoding="utf-8"), f.name)
        return n

    def search(self, query: str, k: int = config.TOP_K) -> list[dict]:
        count = self.col.count()
        if count == 0:
            return []
        res = self.col.query(query_embeddings=self.embedder.embed([query]),
                             n_results=min(k, count))
        return [{"text": d, "source": m["source"], "score": 1 - dist}
                for d, m, dist in zip(res["documents"][0], res["metadatas"][0],
                                      res["distances"][0])]
