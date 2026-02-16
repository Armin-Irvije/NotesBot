"""
Examples: Different ways to use the RAG Chain
"""

from rag_chain import LocalRAGChain, ask_question


# ============================================================================
# EXAMPLE 1: Interactive Chat Mode (Recommended for testing)
# ============================================================================
def example_interactive():
    """Start an interactive chat session"""
    print("Example 1: Interactive Mode")
    print("-" * 60)
    
    rag = LocalRAGChain(
        vector_store_path="./vector_store",
        model_name="llama3",
        temperature=0.7,
        top_k=4
    )
    
    rag.interactive_mode()


# ============================================================================
# EXAMPLE 2: Single Question
# ============================================================================
def example_single_question():
    """Ask a single question"""
    print("Example 2: Single Question")
    print("-" * 60)
    
    rag = LocalRAGChain(vector_store_path="./vector_store")
    
    response = rag.query(
        question="What is RAG evaluation?",
        show_sources=True
    )
    
    print(f"\nFull response object:")
    print(f"Result: {response['result']}")
    print(f"Number of sources: {len(response['source_documents'])}")


# ============================================================================
# EXAMPLE 3: Batch Questions
# ============================================================================
def example_batch_questions():
    """Ask multiple questions in sequence"""
    print("Example 3: Batch Questions")
    print("-" * 60)
    
    rag = LocalRAGChain(vector_store_path="./vector_store")
    
    questions = [
        "What is RAG?",
        "How does embedding work?",
        "What are the main concepts in machine learning?"
    ]
    
    for q in questions:
        print("\n" + "="*60)
        rag.query(q, show_sources=False)


# ============================================================================
# EXAMPLE 4: Using Different Models
# ============================================================================
def example_different_models():
    """Compare answers from different Ollama models"""
    print("Example 4: Different Models")
    print("-" * 60)
    
    question = "Explain transformers in simple terms"
    
    models = ["llama3", "mistral", "phi3"]
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model.upper()}")
        print(f"{'='*60}")
        
        try:
            rag = LocalRAGChain(
                vector_store_path="./vector_store",
                model_name=model
            )
            rag.query(question, show_sources=False)
        except Exception as e:
            print(f"❌ Model {model} not available. Pull it with: ollama pull {model}")


# ============================================================================
# EXAMPLE 5: Adjusting Retrieval Parameters
# ============================================================================
def example_retrieval_parameters():
    """Test different retrieval settings"""
    print("Example 5: Retrieval Parameters")
    print("-" * 60)
    
    question = "What is supervised learning?"
    
    # Try with different numbers of retrieved chunks
    for k in [2, 4, 6]:
        print(f"\n{'='*60}")
        print(f"Retrieving top {k} chunks")
        print(f"{'='*60}")
        
        rag = LocalRAGChain(
            vector_store_path="./vector_store",
            top_k=k
        )
        rag.query(question, show_sources=True)


# ============================================================================
# EXAMPLE 6: Temperature Testing
# ============================================================================
def example_temperature():
    """Test different temperature settings"""
    print("Example 6: Temperature Settings")
    print("-" * 60)
    
    question = "Explain neural networks"
    
    temperatures = [0.0, 0.5, 1.0]
    
    for temp in temperatures:
        print(f"\n{'='*60}")
        print(f"Temperature: {temp} ({'focused' if temp < 0.5 else 'creative'})")
        print(f"{'='*60}")
        
        rag = LocalRAGChain(
            vector_store_path="./vector_store",
            temperature=temp
        )
        rag.query(question, show_sources=False)


# ============================================================================
# EXAMPLE 7: Simple Function Call
# ============================================================================
def example_simple_function():
    """Use the simple ask_question function"""
    print("Example 7: Simple Function Call")
    print("-" * 60)
    
    response = ask_question(
        question="What are the key topics in my notes?",
        vector_store_path="./vector_store",
        model="llama3",
        show_sources=True
    )


# ============================================================================
# EXAMPLE 8: Building a Study Helper
# ============================================================================
def example_study_helper():
    """Use RAG for studying - ask related questions"""
    print("Example 8: Study Helper")
    print("-" * 60)
    
    rag = LocalRAGChain(vector_store_path="./vector_store")
    
    print("\n📚 STUDY SESSION: Machine Learning Basics\n")
    
    study_questions = [
        "What is machine learning?",
        "Can you give me examples from my notes?",
        "What are the main algorithms mentioned?",
        "What should I focus on for the exam?"
    ]
    
    for i, question in enumerate(study_questions, 1):
        print(f"\n{'='*60}")
        print(f"Study Question {i}/{len(study_questions)}")
        print(f"{'='*60}")
        rag.query(question, show_sources=False)
        input("\n[Press Enter for next question...]")


# ============================================================================
# Main Menu
# ============================================================================
def main():
    print("="*60)
    print("RAG CHAIN EXAMPLES")
    print("="*60)
    print("\nChoose an example to run:")
    print("  1. Interactive Mode (recommended)")
    print("  2. Single Question")
    print("  3. Batch Questions")
    print("  4. Different Models Comparison")
    print("  5. Retrieval Parameters Testing")
    print("  6. Temperature Testing")
    print("  7. Simple Function Call")
    print("  8. Study Helper")
    print("\n  0. Exit")
    
    examples = {
        "1": example_interactive,
        "2": example_single_question,
        "3": example_batch_questions,
        "4": example_different_models,
        "5": example_retrieval_parameters,
        "6": example_temperature,
        "7": example_simple_function,
        "8": example_study_helper
    }
    
    choice = input("\nEnter choice (1-8): ").strip()
    
    if choice == "0":
        print("Goodbye!")
        return
    
    if choice in examples:
        print("\n" + "="*60)
        examples[choice]()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
