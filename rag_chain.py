"""
RAG Chain Implementation using FAISS Vector Store and Ollama (Free & Open Source)
No paid APIs required - everything runs locally!
"""

import re
from typing import Any, Dict, List, Optional, Set

from langchain_community.embeddings import OllamaEmbeddings  # pyright: ignore[reportMissingImports]
from langchain_community.vectorstores import FAISS  # pyright: ignore[reportMissingImports]
from langchain_ollama import OllamaLLM  # pyright: ignore[reportMissingImports]
from langchain_classic.chains import RetrievalQA  # pyright: ignore[reportMissingImports]
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

try:
    from sentence_transformers import CrossEncoder  # pyright: ignore[reportMissingImports]
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False


def _tokenize_for_keyword(text: str, min_length: int = 2) -> List[str]:
    """Lowercase alphanumeric tokens, at least min_length chars (ignore pure numbers if desired)."""
    tokens = re.findall(r"\b[a-zA-Z0-9_]{" + str(min_length) + r",}\b", text.lower())
    return tokens


def _build_keyword_index_from_faiss(vector_store: FAISS) -> Dict[str, Set[int]]:
    """
    Build inverted index: term -> set of chunk_index.
    Only chunks with metadata['chunk_index'] are indexed (from build_vectorstore with chunk_index).
    """
    inverted: Dict[str, Set[int]] = {}
    if not hasattr(vector_store, "index_to_docstore_id") or not hasattr(vector_store, "docstore") or not hasattr(vector_store, "index"):
        return inverted
    n = vector_store.index.ntotal
    id_list = vector_store.index_to_docstore_id
    docstore = vector_store.docstore
    for idx in range(n):
        try:
            doc_id = id_list[idx]
            doc = docstore.search(doc_id)
            chunk_index = doc.metadata.get("chunk_index", idx)
            for term in _tokenize_for_keyword(doc.page_content or ""):
                inverted.setdefault(term, set()).add(chunk_index)
        except Exception:
            continue
    return inverted


class SimilaritySearchWithScoreRetriever(BaseRetriever):
    """
    Retriever that uses the vector store's similarity_search_with_score(query, k, filter, **kwargs).
    Optionally filters out documents below a score threshold and attaches score to metadata.

    Direct usage of similarity_search_with_score (without this retriever):
        results = vector_store.similarity_search_with_score(
            query="What is RAG?",
            k=4,
            filter={"source": "mynotes/foo.pdf"},  # optional
        )
        # results: list of (Document, float) tuples; FAISS returns L2 distance (lower = more similar)
    """

    vector_store: VectorStore
    k: int = 4
    metadata_filter: Optional[Dict[str, Any]] = None
    score_threshold: Optional[float] = None
    """If set, only return docs with score >= this (for similarity) or <= this (for distance). FAISS uses L2 distance (lower = better)."""

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # similarity_search_with_score(query, k=4, filter=None, **kwargs)
        results = self.vector_store.similarity_search_with_score(query=query, k=self.k, filter=self.metadata_filter)
        docs: List[Document] = []
        for doc, score in results:
            if self.score_threshold is not None:
                # FAISS returns L2 distance: lower is more similar. So we keep docs with distance <= threshold.
                if score > self.score_threshold:
                    continue
            doc.metadata["retrieval_score"] = score
            docs.append(doc)
        return docs


class HybridKeywordVectorRetriever(BaseRetriever):
    """
    Keyword search before vector search: only chunks that contain at least one query term
    are eligible. Vector search runs and results are restricted to those chunks
    (fetch more from vector, filter by keyword hits, then take top_k).
    """

    vector_store: VectorStore
    k: int = 4
    keyword_index: Optional[Dict[str, Set[int]]] = None
    metadata_filter: Optional[Dict[str, Any]] = None
    score_threshold: Optional[float] = None
    vector_fetch_multiple: int = 3

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_terms = _tokenize_for_keyword(query)
        candidate_indices: Optional[Set[int]] = None
        if self.keyword_index and query_terms:
            candidate_indices = set()
            for term in query_terms:
                candidate_indices.update(self.keyword_index.get(term, set()))
            if not candidate_indices:
                candidate_indices = None
        fetch_k = max(self.k * self.vector_fetch_multiple, 50)
        results = self.vector_store.similarity_search_with_score(
            query=query, k=fetch_k, filter=self.metadata_filter
        )
        filtered: List[tuple] = []
        for doc, score in results:
            if self.score_threshold is not None and score > self.score_threshold:
                continue
            if candidate_indices is not None:
                ci = doc.metadata.get("chunk_index")
                if ci is not None and ci not in candidate_indices:
                    continue
            doc.metadata["retrieval_score"] = score
            filtered.append((doc, score))
        docs = [doc for doc, _ in filtered[: self.k]]
        if not docs and results:
            for doc, score in results[: self.k]:
                if self.score_threshold is not None and score > self.score_threshold:
                    continue
                doc.metadata["retrieval_score"] = score
                docs.append(doc)
        return docs


class MMRRetriever(BaseRetriever):
    """Uses FAISS max_marginal_relevance_search for diverse yet relevant results."""

    vector_store: VectorStore
    k: int = 4
    fetch_k: int = 20
    lambda_mult: float = 0.5
    metadata_filter: Optional[Dict[str, Any]] = None

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.vector_store.max_marginal_relevance_search(
            query=query, k=self.k, fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult, filter=self.metadata_filter,
        )


class MultiQueryEnhancedRetriever(BaseRetriever):
    """Generates alternative query phrasings, retrieves for each, and merges results."""

    base_retriever: BaseRetriever
    llm: Any
    num_queries: int = 3

    def _get_relevant_documents(self, query: str) -> List[Document]:
        prompt = (
            f"Generate {self.num_queries - 1} alternative phrasings of this search query. "
            f"Return only the alternative queries, one per line, no numbering or bullets.\n\n"
            f"Original query: {query}"
        )
        response = self.llm.invoke(prompt)
        alt_queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
        alt_queries = alt_queries[: self.num_queries - 1]

        all_queries = [query] + alt_queries

        seen: Set[str] = set()
        merged: List[Document] = []
        for q in all_queries:
            docs = self.base_retriever.invoke(q)
            for doc in docs:
                key = doc.page_content[:200]
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
        return merged


class ReRankRetriever(BaseRetriever):
    """Wraps a base retriever and re-ranks candidates with a cross-encoder model."""

    base_retriever: BaseRetriever
    reranker: Any
    top_n: int = 5

    def _get_relevant_documents(self, query: str) -> List[Document]:
        candidates = self.base_retriever.invoke(query)
        if not candidates or self.reranker is None:
            return candidates[: self.top_n]

        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)

        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        reranked: List[Document] = []
        for doc, score in scored[: self.top_n]:
            doc.metadata["rerank_score"] = float(score)
            reranked.append(doc)
        return reranked


class LocalRAGChain:
    """RAG Chain using local Ollama models - completely free and open source"""

    def __init__(
        self,
        vector_store_path: str = "./vector_store",
        model_name: str = "llama3",
        embedding_model: str = "nomic-embed-text",
        temperature: float = 0.4,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        use_hybrid_keyword: bool = False,
        use_multi_query: bool = False,
        use_reranker: bool = False,
        rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_mmr: bool = False,
        mmr_lambda: float = 0.5,
    ):
        """
        Initialize the RAG chain with local models

        Args:
            vector_store_path: Path to the FAISS vector store
            model_name: Ollama generative model for answering (llama3, mistral, phi3, etc.)
            embedding_model: Ollama embedding model for vector search (must match the model used to build the vector store)
            temperature: Creativity level (0.0 = focused, 1.0 = creative)
            top_k: Number of relevant chunks to return to the LLM
            score_threshold: Optional max L2 distance for FAISS (lower = more similar); docs with score > this are excluded
            metadata_filter: Optional metadata filter for similarity_search_with_score (e.g. {"source": "mynotes/foo.pdf"})
            use_hybrid_keyword: If True, only chunks containing at least one query term are eligible (keyword before vector)
            use_multi_query: If True, generate alternative query phrasings and merge retrieval results
            use_reranker: If True, retrieve more candidates and re-rank with a cross-encoder (requires sentence-transformers)
            rerank_model_name: HuggingFace cross-encoder model name for re-ranking
            use_mmr: If True, use Maximal Marginal Relevance instead of pure similarity for diverse results
            mmr_lambda: Diversity vs relevance trade-off for MMR (0 = max diversity, 1 = max relevance)
        """
        self.vector_store_path = vector_store_path
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.metadata_filter = metadata_filter
        self.use_hybrid_keyword = use_hybrid_keyword
        self.use_multi_query = use_multi_query
        self.use_reranker = use_reranker
        self.rerank_model_name = rerank_model_name
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.keyword_index: Optional[Dict[str, Set[int]]] = None

        print(f"Initializing Local RAG Chain...")
        print(f"LLM: {model_name} | Embeddings: {embedding_model} (via Ollama)")
        print(f"Vector Store: {vector_store_path}")
        features = []
        if use_hybrid_keyword:
            features.append("hybrid-keyword")
        if use_multi_query:
            features.append("multi-query")
        if use_reranker:
            features.append("reranker")
        if use_mmr:
            features.append("MMR")
        if features:
            print(f"Retrieval features: {', '.join(features)}")

        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vector_store = self._load_vector_store()
        if use_hybrid_keyword and isinstance(self.vector_store, FAISS):
            self.keyword_index = _build_keyword_index_from_faiss(self.vector_store)
            print(f"✓ Keyword index built ({len(self.keyword_index)} terms)")

        self.reranker_model = None
        if use_reranker:
            if not HAS_CROSS_ENCODER:
                print("WARNING: sentence-transformers not installed — re-ranking disabled. "
                      "Install with: pip install sentence-transformers")
                self.use_reranker = False
            else:
                print(f"Loading re-ranker model: {rerank_model_name}...")
                self.reranker_model = CrossEncoder(rerank_model_name)
                print("✓ Re-ranker loaded")

        self.llm = self._initialize_llm()
        self.qa_chain = self._create_qa_chain()
        print("✓ RAG Chain initialized successfully!\n")
    
    def _load_vector_store(self) -> FAISS:
        """Load the FAISS vector store"""
        print("Loading vector store...")
        vector_store = FAISS.load_local(
            self.vector_store_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✓ Loaded {vector_store.index.ntotal} vectors")
        return vector_store
    
    def _initialize_llm(self) -> OllamaLLM:
        """Initialize Ollama LLM with streaming support"""
        print(f"Initializing {self.model_name} model...")
        
        llm = OllamaLLM(
            model=self.model_name,
            temperature=self.temperature,
        )
        
        print(f"✓ {self.model_name} model ready")
        return llm
    
    # defines QA RAG Chain / tooling 
    def _create_qa_chain(self) -> RetrievalQA:
        """Create the retrieval QA chain"""
        
        # Custom prompt template
        prompt_template = """You are a helpful AI assistant answering questions based on the provided context from school notes.

Context from notes:
{context}

Question: {question}

Instructions:
- Answer the question using the information from the context above 
- Answer should be synthesized from the context above into coherent explanation
- If the context doesn't contain enough information to answer, say "I don't have enough information in the notes to answer that question."
- Be concise but thorough
- If relevant, mention which topic or subject the information comes from

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            # this corresponds to the {context} and {question} variables in the prompt template
            input_variables=["context", "question"]
        )
        
        retrieve_k = self.top_k * 3 if self.use_reranker else self.top_k

        if self.use_mmr:
            base_retriever = MMRRetriever(
                vector_store=self.vector_store,
                k=retrieve_k,
                fetch_k=max(retrieve_k * 3, 30),
                lambda_mult=self.mmr_lambda,
                metadata_filter=self.metadata_filter,
            )
        elif self.keyword_index is not None:
            base_retriever = HybridKeywordVectorRetriever(
                vector_store=self.vector_store,
                k=retrieve_k,
                keyword_index=self.keyword_index,
                metadata_filter=self.metadata_filter,
                score_threshold=self.score_threshold,
            )
        else:
            base_retriever = SimilaritySearchWithScoreRetriever(
                vector_store=self.vector_store,
                k=retrieve_k,
                metadata_filter=self.metadata_filter,
                score_threshold=self.score_threshold,
            )

        retriever = base_retriever
        if self.use_multi_query:
            retriever = MultiQueryEnhancedRetriever(
                base_retriever=retriever, llm=self.llm, num_queries=3,
            )
        if self.use_reranker and self.reranker_model is not None:
            retriever = ReRankRetriever(
                base_retriever=retriever, reranker=self.reranker_model, top_n=self.top_k,
            )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )
        
        return qa_chain
    
    def query(self, question: str, show_sources: bool = True) -> dict:
        """
        Ask a question and get an answer from your notes
        
        Args:
            question: The question to ask
            show_sources: Whether to display source documents
            
        Returns:
            Dictionary with 'result' and 'source_documents'
        """
        print(f"\nQuestion: {question}")
        print("-" * 60)
        print("Answer: Thinking...", end="", flush=True)
        
        # Get answer (will stream to console)
        response = self.qa_chain.invoke({"query": question})
        
        print("\n" + "-" * 60)
        
        # Show sources if requested
        if show_sources and response.get("source_documents"):
            print(f"\n📚 Sources ({len(response['source_documents'])} relevant chunks):")
            for i, doc in enumerate(response["source_documents"], 1):
                source = doc.metadata.get("source", "unknown")
                score = doc.metadata.get("retrieval_score")
                score_str = f" (score={score:.4f})" if score is not None else ""
                print(f"\n  [{i}] {source}{score_str}")
                print(f"      {doc.page_content[:150]}...")
                print(response["result"])      

        return response
    
    def interactive_mode(self):
        """Start an interactive Q&A session"""
        print("\n" + "="*60)
        print("🤖 LOCAL RAG CHATBOT - Interactive Mode")
        print("="*60)
        print("Ask questions about your notes. Type 'quit' or 'exit' to stop.\n")
        
        while True:
            try:
                question = input("\n💬 You: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye! 👋")
                    break
                
                if not question:
                    continue
                
                self.query(question, show_sources=True)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


# Alternative: Simple function-based approach
def ask_question(question: str, vector_store_path: str = "./vector_store", model: str = "llama3", embedding_model: str = "nomic-embed-text", show_sources: bool = True):
    """
    Simple function to ask a single question

    Args:
        question: Question to ask
        vector_store_path: Path to vector store
        model: Ollama generative model name
        embedding_model: Ollama embedding model name (must match vector store)
        show_sources: Show source documents
    """
    chain = LocalRAGChain(
        vector_store_path=vector_store_path,
        model_name=model,
        embedding_model=embedding_model,
    )
    return chain.query(question, show_sources=show_sources)


if __name__ == "__main__":
    # Example 1: Interactive mode
    print("Starting RAG Chatbot in Interactive Mode...")
    print("\nMake sure you have:")
    print("  1. Built the vector store (python build_vectorstore.py)")
    print("  2. Ollama running (ollama serve)")
    print("  3. LLM model available (ollama pull llama3)")
    print("  4. Embedding model available (ollama pull nomic-embed-text)")
    print()

    try:
        rag_chain = LocalRAGChain(
            vector_store_path="./vector_store",
            model_name="llama3",
            embedding_model="nomic-embed-text",
            temperature=0.4,
            top_k=5,
        )
        
        # Start interactive mode
        rag_chain.interactive_mode()
        
    except FileNotFoundError:
        print("\n❌ Vector store not found!")
        print("Please run 'python build_vectorstore.py' first.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure Ollama is running: ollama serve")
