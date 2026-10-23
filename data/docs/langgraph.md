LangGraph is a library for building stateful, multi-step LLM applications as graphs. Nodes are Python functions that read and update a shared state, and edges decide which node runs next.

Conditional edges let an agent branch at runtime, for example to retry retrieval with a rewritten query when the retrieved documents are not relevant.
