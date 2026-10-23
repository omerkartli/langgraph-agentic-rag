from app.graph import FALLBACK, build_graph
from app.models import ScriptedLLM


def run(llm, store, q, **kw):
    return build_graph(llm, store, **kw).invoke({"question": q})


def test_small_talk_skips_retrieval(store):
    out = run(ScriptedLLM(), store, "hello")
    assert out["trace"] == ["route", "generate_direct"]


def test_grounded_answer_from_documents(store):
    out = run(ScriptedLLM(), store, "What is LangGraph used for?")
    assert out["trace"] == ["route", "retrieve", "grade", "generate", "check"]
    assert out["grounded"] and out["retries"] == 0
    assert {d["source"] for d in out["documents"]} == {"langgraph.md"}


def test_unanswerable_question_retries_then_falls_back(store):
    out = run(ScriptedLLM(), store, "Who won the 1998 World Cup?", max_retries=2)
    assert out["answer"] == FALLBACK
    assert out["retries"] == 2
    assert out["trace"].count("rewrite") == 2 and out["trace"][-1] == "fallback"


def test_rewrite_recovers_retrieval(store, monkeypatch):
    search = store.search
    monkeypatch.setattr(store, "search", lambda q, k=1: search(q, k=1))  # tiny corpus: force top-1
    llm = ScriptedLLM({
        "rewrite": lambda p: "Cloud Run deploy FastAPI container",
        "grade": lambda p: "yes" if "Cloud Run" in p.split("DOCUMENT:")[1] else "no",
    })
    out = run(llm, store, "zebra quantum ship")
    assert out["retries"] == 1 and out["grounded"]
    assert out["trace"][:5] == ["route", "retrieve", "grade", "rewrite", "retrieve"]
    assert "cloud-run.md" in {d["source"] for d in out["documents"]}


def test_ungrounded_answer_triggers_retry_and_stops_at_limit(store):
    llm = ScriptedLLM({"grounded": lambda p: "no"})
    out = run(llm, store, "What is LangGraph used for?", max_retries=1)
    assert out["trace"].count("check") == 2 and out["retries"] == 1
    assert out["grounded"] is False  # surfaced to the caller, not hidden
