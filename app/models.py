"""LLM and embedding providers behind two tiny interfaces.

`vertex` talks to Gemini / text-embedding on Vertex AI (GCP).
`fake` is a deterministic offline implementation used by tests and local demos,
so the graph logic can be verified without credentials or network access.
"""
import hashlib
import math
import re
from typing import Callable, Protocol

from app import config


class LLM(Protocol):
    def invoke(self, prompt: str) -> str: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


_WORD = re.compile(r"\w+", re.UNICODE)


def tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


class HashEmbedder:
    """Bag-of-words hashing embedder: lexical overlap -> cosine similarity."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in tokens(t):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class ScriptedLLM:
    """Offline LLM: answers by `[TASK:<name>]` tag using a handler per task."""

    def __init__(self, handlers: dict[str, Callable[[str], str]] | None = None):
        self.handlers = handlers or {}
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> str:
        m = re.match(r"\[TASK:(\w+)\]", prompt)
        task = m.group(1) if m else "generate"
        self.calls.append(task)
        return self.handlers.get(task, _default_handler(task))(prompt)


def _field(prompt: str, name: str) -> str:
    m = re.search(rf"{name}:\s*(.*?)(?=\n[A-Z_]+:|\Z)", prompt, re.S)
    return m.group(1).strip() if m else ""


def _overlap(a: str, b: str) -> float:
    ta, tb = set(tokens(a)), set(tokens(b))
    return len(ta & tb) / (len(ta) or 1)


def _default_handler(task: str) -> Callable[[str], str]:
    def route(p: str) -> str:
        q = _field(p, "QUESTION").lower()
        return "direct" if q in {"hi", "hello", "merhaba", "selam", "thanks"} else "retrieve"

    def grade(p: str) -> str:
        return "yes" if _overlap(_field(p, "QUESTION"), _field(p, "DOCUMENT")) >= 0.3 else "no"

    def rewrite(p: str) -> str:
        return _field(p, "QUESTION") + " details"

    def generate(p: str) -> str:
        ctx = _field(p, "CONTEXT")
        return ctx.split("\n")[0] if ctx else "Hello! Ask me about the indexed documents."

    def grounded(p: str) -> str:
        return "yes" if _overlap(_field(p, "ANSWER"), _field(p, "CONTEXT")) >= 0.6 else "no"

    return {"route": route, "grade": grade, "rewrite": rewrite,
            "generate": generate, "grounded": grounded}.get(task, generate)


class VertexLLM:
    def __init__(self):
        from langchain_google_vertexai import ChatVertexAI

        self._chat = ChatVertexAI(
            model_name=config.CHAT_MODEL, project=config.PROJECT or None,
            location=config.LOCATION, temperature=0,
        )

    def invoke(self, prompt: str) -> str:
        return str(self._chat.invoke(prompt).content)


class VertexEmbedder:
    def __init__(self):
        from langchain_google_vertexai import VertexAIEmbeddings

        self._emb = VertexAIEmbeddings(
            model_name=config.EMBED_MODEL, project=config.PROJECT or None,
            location=config.LOCATION,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._emb.embed_documents(texts)


def get_llm() -> LLM:
    return VertexLLM() if config.LLM_PROVIDER == "vertex" else ScriptedLLM()


def get_embedder() -> Embedder:
    return VertexEmbedder() if config.LLM_PROVIDER == "vertex" else HashEmbedder()
