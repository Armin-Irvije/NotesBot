"""
RAG Chain Implementation using FAISS Vector Store and Ollama (Free & Open Source)
No paid APIs required - everything runs locally!
"""

from typing import Any, Dict, List, Optional

from langchain_community.embeddings import OllamaEmbeddings  # pyright: ignore[reportMissingImports]
from langchain_community.vectorstores import FAISS  # pyright: ignore[reportMissingImports]
from langchain_ollama import OllamaLLM  # pyright: ignore[reportMissingImports]
from langchain_classic.chains import RetrievalQA  # pyright: ignore[reportMissingImports]
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore


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


class LocalRAGChain:
    """RAG Chain using local Ollama models - completely free and open source"""
    # Default values
    def __init__(
        self,
        vector_store_path: str = "./vector_store",
        model_name: str = "llama3",
        temperature: float = 0.4,
        top_k: int = 3,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the RAG chain with local models

        Args:
            vector_store_path: Path to the FAISS vector store
            model_name: Ollama model to use (llama3, mistral, phi3, etc.)
            temperature: Creativity level (0.0 = focused, 1.0 = creative)
            top_k: Number of relevant chunks to retrieve (passed as k to similarity_search_with_score)
            score_threshold: Optional max L2 distance for FAISS (lower = more similar); docs with score > this are excluded
            metadata_filter: Optional metadata filter for similarity_search_with_score (e.g. {"source": "mynotes/foo.pdf"})
        """
        self.vector_store_path = vector_store_path
        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.metadata_filter = metadata_filter
        
        print(f"Initializing Local RAG Chain...")
        print(f"Model: {model_name} (via Ollama)")
        print(f"Vector Store: {vector_store_path}")
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=model_name)
        
        # Load vector store
        self.vector_store = self._load_vector_store()
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Create retrieval chain
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
        
        # Retriever that uses similarity_search_with_score(query, k, filter, **kwargs)
        retriever = SimilaritySearchWithScoreRetriever(
            vector_store=self.vector_store,
            k=self.top_k,
            metadata_filter=self.metadata_filter,
            score_threshold=self.score_threshold,
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
def ask_question(question: str, vector_store_path: str = "./vector_store", model: str = "llama3", show_sources: bool = True):
    """
    Simple function to ask a single question
    
    Args:
        question: Question to ask
        vector_store_path: Path to vector store
        model: Ollama model name
        show_sources: Show source documents
    """
    chain = LocalRAGChain(
        vector_store_path=vector_store_path,
        model_name=model
    )
    return chain.query(question, show_sources=show_sources)


if __name__ == "__main__":
    # Example 1: Interactive mode
    print("Starting RAG Chatbot in Interactive Mode...")
    print("\nMake sure you have:")
    print("  1. Built the vector store (python build_vectorstore.py)")
    print("  2. Ollama running (ollama serve)")
    print("  3. Model available (ollama pull llama3)")
    print()
    
    try:
        # Initialize RAG chain
        rag_chain = LocalRAGChain(
            vector_store_path="./vector_store",
            model_name="llama3",  # Change to: mistral, phi3, gemma, etc.
            temperature=0.4,
            top_k=5
        )
        
        # Start interactive mode
        rag_chain.interactive_mode()
        
    except FileNotFoundError:
        print("\n❌ Vector store not found!")
        print("Please run 'python build_vectorstore.py' first.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure Ollama is running: ollama serve")
