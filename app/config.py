import os

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "fake")
PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "europe-west4")
CHAT_MODEL = os.getenv("VERTEX_CHAT_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("VERTEX_EMBED_MODEL", "text-embedding-005")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
TOP_K = int(os.getenv("TOP_K", "4"))
DOCS_DIR = os.getenv("DOCS_DIR", "data/docs")
