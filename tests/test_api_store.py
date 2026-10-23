from fastapi.testclient import TestClient

from app import api
from app.store import chunk


def test_chunk_overlap_and_size():
    text = "\n\n".join(f"para {i} " + "x" * 200 for i in range(6))
    parts = chunk(text, size=450, overlap=50)
    assert len(parts) > 1 and all(len(p) < 800 for p in parts)


def test_search_returns_best_source_first(store):
    hits = store.search("Chroma HNSW cosine distance")
    assert hits[0]["source"] == "chroma.md"


def test_api_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "c"))
    monkeypatch.setattr("app.config.CHROMA_PATH", str(tmp_path / "c"))
    with TestClient(api.app) as client:
        assert client.get("/health").json()["chunks"] >= 3
        r = client.post("/ask", json={"question": "What is Cloud Run?"}).json()
        assert r["grounded"] and "cloud-run.md" in r["sources"]
