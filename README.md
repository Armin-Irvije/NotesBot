# NotesBot (Local RAG Playground)

This project is my hands-on attempt at building a **RAG (Retrieval-Augmented Generation)** system end-to-end using **local Ollama models**, mainly so I could understand the moving pieces for myself rather than treating RAG as a black box.

## Why I built this

I wanted to see, in code, how different choices affect answer quality and failure modes:

- How **chunking** decisions change what gets retrieved and what context the model sees
- How **vectorization/embeddings** influence semantic matching
- How retrieval quality changes when you mix **lexical keyword matching** with **semantic search**
- How much **re-ranking** can improve “pretty good” retrieval into “actually useful” retrieval
- How query strategies like **multi-query** and **maximal marginal relevance (MMR)** affect recall, diversity, and redundancy

## What I implemented

At a high level, the system builds a vector store from my notes and answers questions by retrieving relevant chunks and passing them to a local LLM.

Retrieval experimentation includes:

- **Semantic retrieval** using vector similarity search
- **Hybrid retrieval** combining keyword-style indexing with vector search to narrow candidates
- **Re-ranking** of retrieved candidates using a cross-encoder model to improve ordering
- **Multi-query retrieval** (generate alternative phrasings, retrieve per query, then merge results)
- **MMR retrieval** to balance relevance with diversity and reduce near-duplicate chunks in the final context

Overall, this repo is less about being a polished product and more about being a **learning-oriented RAG implementation** where I can inspect and iterate on each stage of the pipeline.

