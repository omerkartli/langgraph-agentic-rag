"""python -m app.ingest data/docs"""
import sys

from app.models import get_embedder
from app.store import VectorStore

if __name__ == "__main__":
    n = VectorStore(get_embedder()).ingest_dir(sys.argv[1] if len(sys.argv) > 1 else "data/docs")
    print(f"Indexed {n} chunks")
