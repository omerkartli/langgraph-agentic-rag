"""Agentic RAG as a LangGraph state machine.

    route ─┬─ direct ──────────────────────────────► generate_direct ─► END
           └─ retrieve ─► grade ─┬─ relevant ─► generate ─► check ─┬─ grounded ─► END
                 ▲               │                                  └─ not grounded ─┐
                 │               └─ none relevant ─┐                                  │
                 └──────────── rewrite ◄───────────┴──────── (while retries left) ◄──┘
                                   └─ retries exhausted ─► fallback / flagged answer
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app import config
from app.models import LLM
from app.store import VectorStore

FALLBACK = "I could not find a reliable answer in the indexed documents."


class RAGState(TypedDict, total=False):
    question: str        # original user question
    query: str           # current (possibly rewritten) retrieval query
    route: str
    documents: list[dict]
    answer: str
    grounded: bool
    retries: int
    trace: list[str]     # visited nodes, returned to the caller for transparency


def build_graph(llm: LLM, store: VectorStore, max_retries: int = config.MAX_RETRIES):
    def log(state: RAGState, node: str) -> list[str]:
        return state.get("trace", []) + [node]

    def route(state: RAGState) -> RAGState:
        out = llm.invoke(f"[TASK:route]\nDecide if the question needs document retrieval "
                         f"or is small talk. Reply 'retrieve' or 'direct'.\n"
                         f"QUESTION: {state['question']}").strip().lower()
        return {"route": "direct" if out.startswith("direct") else "retrieve",
                "query": state["question"], "retries": 0, "trace": log(state, "route")}

    def generate_direct(state: RAGState) -> RAGState:
        ans = llm.invoke(f"[TASK:generate]\nReply briefly.\nQUESTION: {state['question']}\nCONTEXT:")
        return {"answer": ans, "grounded": True, "trace": log(state, "generate_direct")}

    def retrieve(state: RAGState) -> RAGState:
        return {"documents": store.search(state["query"]), "trace": log(state, "retrieve")}

    def grade(state: RAGState) -> RAGState:
        keep = [d for d in state.get("documents", []) if llm.invoke(
            f"[TASK:grade]\nIs the document relevant to the question? Reply 'yes' or 'no'.\n"
            f"QUESTION: {state['question']}\nDOCUMENT: {d['text']}").strip().lower().startswith("yes")]
        return {"documents": keep, "trace": log(state, "grade")}

    def rewrite(state: RAGState) -> RAGState:
        new_q = llm.invoke(f"[TASK:rewrite]\nRewrite the question as a better search query.\n"
                           f"QUESTION: {state['query']}").strip()
        return {"query": new_q or state["query"], "retries": state.get("retries", 0) + 1,
                "trace": log(state, "rewrite")}

    def generate(state: RAGState) -> RAGState:
        ctx = "\n".join(d["text"] for d in state["documents"])
        ans = llm.invoke(f"[TASK:generate]\nAnswer using ONLY the context.\n"
                         f"QUESTION: {state['question']}\nCONTEXT: {ctx}")
        return {"answer": ans.strip(), "trace": log(state, "generate")}

    def check(state: RAGState) -> RAGState:
        ctx = "\n".join(d["text"] for d in state["documents"])
        verdict = llm.invoke(f"[TASK:grounded]\nIs every claim in the answer supported by the "
                             f"context? Reply 'yes' or 'no'.\nANSWER: {state['answer']}\n"
                             f"CONTEXT: {ctx}").strip().lower()
        return {"grounded": verdict.startswith("yes"), "trace": log(state, "check")}

    def fallback(state: RAGState) -> RAGState:
        return {"answer": FALLBACK, "grounded": False, "documents": [],
                "trace": log(state, "fallback")}

    def after_route(state: RAGState) -> str:
        return state["route"]

    def after_grade(state: RAGState) -> str:
        if state["documents"]:
            return "generate"
        return "rewrite" if state["retries"] < max_retries else "fallback"

    def after_check(state: RAGState) -> str:
        if state["grounded"]:
            return "done"
        return "rewrite" if state["retries"] < max_retries else "done"

    g = StateGraph(RAGState)
    for name, fn in [("route", route), ("generate_direct", generate_direct),
                     ("retrieve", retrieve), ("grade", grade), ("rewrite", rewrite),
                     ("generate", generate), ("check", check), ("fallback", fallback)]:
        g.add_node(name, fn)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route,
                            {"direct": "generate_direct", "retrieve": "retrieve"})
    g.add_edge("generate_direct", END)
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade,
                            {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "check")
    g.add_conditional_edges("check", after_check, {"done": END, "rewrite": "rewrite"})
    g.add_edge("fallback", END)
    return g.compile()
