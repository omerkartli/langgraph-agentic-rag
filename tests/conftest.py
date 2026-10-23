import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import HashEmbedder  # noqa: E402
from app.store import VectorStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    s = VectorStore(HashEmbedder(), path=str(tmp_path / "chroma"))
    s.ingest_dir(str(Path(__file__).resolve().parents[1] / "data" / "docs"))
    return s
