# Phase Two — Agent Orchestration & Vector Retrieval (Onboarding)

## Purpose
This document gives a concise architectural and code-level introduction focused on:
- Agent logic and tool orchestration
- Vector retrieval and document search

It complements the existing `README.md` and points you to the important files to inspect next.

---

## 1. High-level flow (quick)

User Query → FastAPI route / WebSocket → Build/Load Agent → Agent reasons & calls Tools → Tools query FAISS retrievers → Agent synthesizes → Response streamed back

---

## 2. Key files to know

- `main.py` — API & WebSocket endpoints, session handling.
- `agent_loader.py` — Builds agents (system + firebase), collects tools, creates LangChain agents.
- `agent.py` — Backwards-compatible init wrapper to pick agent by database.
- `llm.py` — Azure OpenAI client/embeddings initialization.
- `vectorstores.py` — FAISS index loading helpers.
- `retriever_*.py` (e.g., `retriever_tools.py`, `retriever_parent.py`, `retriever_vectorstores.py`) — The various retrieval strategies and tool implementations.

---

## 3. Agent orchestration (architecture + code pointers)

What is an agent?
- An agent is a LangChain construct that receives user input, decides which tools to call, executes them, observes results, and returns a final answer.

How agents are configured:
- System manifests in `agent_loader.py` (e.g., `SYSTEM_AGENT_CONFIGS`) define `tool_groups`, `internal_guidelines` (system prompt), `database`, and other metadata.

Example config (conceptual):
```py
SYSTEM_AGENT_CONFIGS = {
  "juris": {
    "id": "juris",
    "tool_groups": ["rets_base"],
    "internal_guidelines": system_prompt_string,
  }
}
```

Agent build flow (high-level):
1. Load config (system or Firebase)
2. Collect tools via `_collect_tool_objects(tool_groups)`
3. Optionally add user-upload tools (if `uid` present)
4. Construct LangChain agent with `create_conversational_tools_agent(llm, tools, system_prompt)`

Code to inspect:
- `agent_loader.py` — search for `_collect_tool_objects`, `_create_user_uploads_tool`, and the function that returns the actual agent instance.

---

## 4. Tools & orchestration pattern

What is a Tool?
- A tool is a callable decorated or wrapped for LangChain (often `@tool`) that performs a specialized action (e.g., search a FAISS index, format results, call external API).

Typical tool pattern (from `retriever_tools.py`):
```py
@tool("rets_get_correct_shortNames")
def rets_get_correct_shortNames(question: str) -> str:
    # uses multi retriever
    results = get_db_main_rets().similarity_search(query=question, k=100)
    return formatted_text
```

ReAct pattern in practice:
- Agent reasons: which tool to call.
- Calls tool that returns structured text or snippets.
- Agent may call further tools using tool output.
- Final synthesis uses LLM to produce the answer.

Inspect these in the repo:
- `retriever_tools.py` (many tools implemented)
- `agent_loader.py` (how tools are attached to agents)

---

## 5. Vector retrieval & FAISS (how search works)

FAISS Indexes:
- Pre-built FAISS indexes live under folders like `db_main_rets`, `db_ast_dk`, etc.
- `vectorstores.py` contains lazy loaders (e.g., `get_db_rets()` and `get_db_ast()`) that call `FAISS.load_local(path, get_embeddings())`.

Embeddings:
- `llm.py` provides `get_embeddings()` which returns an Azure embeddings client (deployment `Propria-embed-3-small` by default).

Retrieval strategies (why multiple):
- Basic similarity search (nearest neighbors)
- Parent document retrieval (get whole doc + important chunks)
- Multi-query / query rewriting (search multiple reformulations)
- Contextual compression (extract only relevant paragraphs)

Typical similarity call:
```py
results = get_db_main_rets().similarity_search(query=question, k=50, filter={"shortName": shortName})
```

Metadata returned with `doc` objects commonly includes `title`, `shortName`, `source`, and `page_content`.

---

## 6. User uploads & shared agent uploads (brief)

- Uploaded files are indexed into FAISS via Azure Blob pipelines (`azure_user_uploads.py`, `azure_agent_uploads.py`).
- Agent loader supports adding a `user_uploads_search` tool that queries per-user or per-agent upload FAISS indexes.
- There is caching and ETag checks to avoid reloading indexes per request.

Files: inspect `_create_user_uploads_tool` and `_load_shared_agent_uploads_index` in `agent_loader.py`.

---

## 7. Where to look first (recommended path for a beginner)

1. Open `backend/main.py` and find the WebSocket or chat route to see how requests enter the system.
2. Open `backend/agent.py` and `backend/agent_loader.py` to follow how a selected database → agent_id becomes a runnable agent.
3. Browse `backend/retriever_tools.py` to see concrete tool implementations and how FAISS results are formatted.
4. Inspect `backend/vectorstores.py` and `backend/llm.py` to understand index loading and embeddings.

---

