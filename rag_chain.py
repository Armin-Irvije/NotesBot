"""
RAG Chain Implementation using FAISS Vector Store and Ollama (Free & Open Source)
No paid APIs required - everything runs locally!
"""

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class LocalRAGChain:
    """RAG Chain using local Ollama models - completely free and open source"""
    
    def __init__(
        self,
        vector_store_path: str = "./vector_store",
        model_name: str = "llama3",
        temperature: float = 0.7,
        top_k: int = 3
    ):
        """
        Initialize the RAG chain with local models
        
        Args:
            vector_store_path: Path to the FAISS vector store
            model_name: Ollama model to use (llama3, mistral, phi3, etc.)
            temperature: Creativity level (0.0 = focused, 1.0 = creative)
            top_k: Number of relevant chunks to retrieve
        """
        self.vector_store_path = vector_store_path
        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k
        
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
- Answer the question using the information from the context above and synthesis a clear and concise answer
- If the context doesn't contain enough information to answer, say "I don't have enough information in the notes to answer that question."
- Be concise but thorough
- If relevant, mention which topic or subject the information comes from

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            # this corresponds to the {context} and {question} variables in the prompt template
            input_variables=["context", "question"]
        )
        
        # Create retrieval chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": self.top_k}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
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
        print("Answer: ", end="", flush=True)
        
        # Get answer (will stream to console)
        response = self.qa_chain.invoke({"query": question})
        
        print("\n" + "-" * 60)
        
        # Show sources if requested
        if show_sources and response.get("source_documents"):
            print(f"\n📚 Sources ({len(response['source_documents'])} relevant chunks):")
            for i, doc in enumerate(response["source_documents"], 1):
                source = doc.metadata.get("source", "unknown")
                print(f"\n  [{i}] {source}")
                print(f"      {doc.page_content[:150]}...")
        
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
def ask_question(
    question: str,
    vector_store_path: str = "./vector_store",
    model: str = "llama3",
    show_sources: bool = True
):
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
            temperature=0.7,
            top_k=3
        )
        
        # Start interactive mode
        rag_chain.interactive_mode()
        
    except FileNotFoundError:
        print("\n❌ Vector store not found!")
        print("Please run 'python build_vectorstore.py' first.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure Ollama is running: ollama serve")
