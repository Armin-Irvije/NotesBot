"""
Chunking and Embedding Script for LangChain RAG Chatbot
Splits documents, generates embeddings, and stores in FAISS vector database
"""

import os
import pickle
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from document_loader import NotesLoader


class VectorStoreBuilder:
    """Build and manage FAISS vector store from documents"""
    
    def __init__(
        self,
        chunk_size: int = 1000, # characters not words
        chunk_overlap: int = 200, # over lap means some chunks have the same text
        embedding_model: str = "llama3"
    ):
        """
        Initialize the vector store builder
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
            embedding_model: Ollama model to use for embeddings
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize embeddings
        print(f"Initializing Ollama embeddings with model: {self.embedding_model}")
        self.embeddings = OllamaEmbeddings(model=self.embedding_model) # langchain community package for ollama embeddings
        # Juris legal uses Open AI propria-3-embed model 
        
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        print(f"\nSplitting {len(documents)} documents into chunks...")
        print(f"Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        
        chunks = self.text_splitter.split_documents(documents)
        
        print(f"✓ Created {len(chunks)} chunks from {len(documents)} documents")
        print(f"  Average chunks per document: {len(chunks) / len(documents):.1f}")
        
        return chunks
    
    def create_vector_store(self, chunks: List[Document]) -> FAISS:
        """
        Create FAISS vector store from document chunks
        
        Args:
            chunks: List of document chunks
            
        Returns:
            FAISS vector store
        """
        print(f"\nGenerating embeddings for {len(chunks)} chunks...")
        print("This may take a few minutes depending on the number of chunks...")
        
        # Create vector store
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        print(f"✓ Vector store created successfully!")
        print(f"  Total vectors: {vector_store.index.ntotal}")
        
        return vector_store
    
    def save_vector_store(
        self,
        vector_store: FAISS,
        save_path: str = "./vector_store"
    ):
        """
        Save vector store to disk for reuse
        
        Args:
            vector_store: FAISS vector store to save
            save_path: Directory path to save the vector store
        """
        print(f"\nSaving vector store to: {save_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save FAISS index
        vector_store.save_local(save_path)
        
        print(f"✓ Vector store saved successfully!")
        print(f"  Location: {os.path.abspath(save_path)}")
    
    def load_vector_store(
        self,
        load_path: str = "./vector_store"
    ) -> FAISS:
        """
        Load existing vector store from disk
        
        Args:
            load_path: Directory path to load the vector store from
            
        Returns:
            Loaded FAISS vector store
        """
        print(f"\nLoading vector store from: {load_path}")
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Vector store not found at: {load_path}")
        
        vector_store = FAISS.load_local(
            load_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        print(f"✓ Vector store loaded successfully!")
        print(f"  Total vectors: {vector_store.index.ntotal}")
        
        return vector_store
    
    def test_similarity_search(
        self,
        vector_store: FAISS,
        query: str,
        k: int = 3
    ):
        """
        Test the vector store with a sample query
        
        Args:
            vector_store: FAISS vector store
            query: Test query string
            k: Number of results to return
        """
        print(f"\n{'='*60}")
        print(f"TESTING SIMILARITY SEARCH")
        print(f"{'='*60}")
        print(f"Query: '{query}'")
        print(f"Retrieving top {k} most relevant chunks...\n")
        
        results = vector_store.similarity_search(query, k=k)
        
        for i, doc in enumerate(results, 1):
            print(f"--- Result {i} ---")
            print(f"Source: {doc.metadata.get('source', 'unknown')}")
            print(f"Content: {doc.page_content[:200]}...")
            print()


def build_vector_database(
    notes_directory: str = "./mynotes",
    vector_store_path: str = "./vector_store",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    test_query: str = "What is RAG?"
):
    """
    Main function to build the complete vector database
    
    Args:
        notes_directory: Path to notes folder
        vector_store_path: Path to save vector store
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        test_query: Query to test the vector store
    """
    print("="*60)
    print("BUILDING VECTOR DATABASE FOR RAG CHATBOT")
    print("="*60)
    
    # Step 1: Load documents
    print("\n[STEP 1/4] Loading documents...")
    loader = NotesLoader(notes_directory=notes_directory)
    documents = loader.load_documents()
    
    if not documents:
        print("❌ No documents found! Please add files to the mynotes folder.")
        return
    
    # Step 2: Initialize builder
    print("\n[STEP 2/4] Initializing vector store builder...")
    builder = VectorStoreBuilder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model="llama3"
    )
    
    # Step 3: Split into chunks
    print("\n[STEP 3/4] Splitting documents into chunks...")
    chunks = builder.split_documents(documents)
    
    # Step 4: Create and save vector store
    print("\n[STEP 4/4] Creating vector store with embeddings...")
    vector_store = builder.create_vector_store(chunks)
    
    # Save vector store
    builder.save_vector_store(vector_store, save_path=vector_store_path)
    
    # Test the vector store
    if test_query:
        builder.test_similarity_search(vector_store, query=test_query, k=3)
    
    print("\n" + "="*60)
    print("✅ VECTOR DATABASE BUILD COMPLETE!")
    print("="*60)
    print(f"Documents processed: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Vector store saved to: {os.path.abspath(vector_store_path)}")
    print("\nYou can now use this vector store for your RAG chatbot!")


if __name__ == "__main__":
    # Build the vector database
    build_vector_database(
        notes_directory="./mynotes",
        vector_store_path="./vector_store",
        chunk_size=1000,
        chunk_overlap=200,
        test_query="What is RAG evaluation?"
    )
