"""
Quick test script to verify vector store is working correctly
Run this after building your vector store
"""

from build_vectorstore import VectorStoreBuilder


def test_vector_store():
    """Test loading and querying the vector store"""
    
    print("Testing Vector Store")
    print("=" * 60)
    
    # Initialize builder
    builder = VectorStoreBuilder(embedding_model="llama3")
    
    try:
        # Load the vector store
        print("\n1. Loading vector store...")
        vector_store = builder.load_vector_store("./vector_store")
        
        # Test queries
        test_queries = [
            "What is RAG evaluation?",
            "machine learning concepts",
            "explain the main topic"
        ]
        
        print("\n2. Running test queries...")
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: '{query}'")
            print(f"{'='*60}")
            
            results = vector_store.similarity_search(query, k=2)
            
            for i, doc in enumerate(results, 1):
                print(f"\n--- Result {i} ---")
                print(f"Source: {doc.metadata.get('source', 'unknown')}")
                print(f"Content (first 300 chars):")
                print(doc.page_content[:300])
                print("...")
        
        print("\n" + "="*60)
        print("✅ Vector store is working correctly!")
        print("="*60)
        
    except FileNotFoundError:
        print("\n❌ Vector store not found!")
        print("Please run 'python build_vectorstore.py' first to create it.")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    test_vector_store()
