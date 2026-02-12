"""
Document Loader for LangChain RAG Chatbot
Loads .txt, .md, and .pdf files from the mynotes folder
"""

import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader
)
from langchain_core.documents import Document


class NotesLoader:
    """Load documents from mynotes folder supporting multiple file types"""
    
    def __init__(self, notes_directory: str = "./mynotes"):
        self.notes_directory = notes_directory
        self.supported_extensions = ['.txt', '.md', '.pdf']
        
    def load_documents(self) -> List[Document]:
        """
        Load all supported documents from the notes directory
        
        Returns:
            List[Document]: List of loaded documents with content and metadata
        """
        all_documents = []
        
        # Check if directory exists
        if not os.path.exists(self.notes_directory):
            raise FileNotFoundError(f"Directory '{self.notes_directory}' not found!")
        
        print(f"Loading documents from: {self.notes_directory}")
        
        # Load .txt files
        txt_docs = self._load_text_files()
        all_documents.extend(txt_docs)
        
        # Load .md (markdown) files
        md_docs = self._load_markdown_files()
        all_documents.extend(md_docs)
        
        # Load .pdf files
        pdf_docs = self._load_pdf_files()
        all_documents.extend(pdf_docs)
        
        print(f"\nTotal documents loaded: {len(all_documents)}")
        return all_documents
    
    def _load_text_files(self) -> List[Document]:
        """Load all .txt files"""
        txt_files = list(Path(self.notes_directory).rglob("*.txt"))
        documents = []
        
        for file_path in txt_files:
            try:
                loader = TextLoader(str(file_path), encoding='utf-8')
                docs = loader.load()
                documents.extend(docs)
                print(f"✓ Loaded: {file_path.name}")
            except Exception as e:
                print(f"✗ Error loading {file_path.name}: {str(e)}")
        
        print(f"Text files loaded: {len(documents)}")
        return documents
    
    def _load_markdown_files(self) -> List[Document]:
        """Load all .md files"""
        md_files = list(Path(self.notes_directory).rglob("*.md"))
        documents = []
        
        for file_path in md_files:
            try:
                loader = TextLoader(str(file_path), encoding='utf-8')
                docs = loader.load()
                documents.extend(docs)
                print(f"✓ Loaded: {file_path.name}")
            except Exception as e:
                print(f"✗ Error loading {file_path.name}: {str(e)}")
        
        print(f"Markdown files loaded: {len(documents)}")
        return documents
    
    def _load_pdf_files(self) -> List[Document]:
        """Load all .pdf files"""
        pdf_files = list(Path(self.notes_directory).rglob("*.pdf"))
        documents = []
        
        for file_path in pdf_files:
            try:
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                documents.extend(docs)
                print(f"✓ Loaded: {file_path.name} ({len(docs)} pages)")
            except Exception as e:
                print(f"✗ Error loading {file_path.name}: {str(e)}")
        
        print(f"PDF files loaded: {len(documents)}")
        return documents
    
    def get_document_stats(self, documents: List[Document]) -> dict:
        """Get statistics about loaded documents"""
        total_chars = sum(len(doc.page_content) for doc in documents)
        
        stats = {
            'total_documents': len(documents),
            'total_characters': total_chars,
            'avg_chars_per_doc': total_chars // len(documents) if documents else 0,
            'sources': list(set(doc.metadata.get('source', 'unknown') for doc in documents))
        }
        return stats


# Example usage and testing
if __name__ == "__main__":
    # Initialize the loader
    loader = NotesLoader(notes_directory="./mynotes")
    
    try:
        # Load all documents
        documents = loader.load_documents()
        
        # Get and print statistics
        stats = loader.get_document_stats(documents)
        print("\n" + "="*50)
        print("DOCUMENT STATISTICS")
        print("="*50)
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Total Characters: {stats['total_characters']:,}")
        print(f"Average Chars/Doc: {stats['avg_chars_per_doc']:,}")
        print(f"\nSources found:")
        for source in stats['sources']:
            print(f"  - {source}")
        
        # Preview first document
        if documents:
            print("\n" + "="*50)
            print("PREVIEW - First Document")
            print("="*50)
            print(f"Source: {documents[0].metadata.get('source', 'unknown')}")
            print(f"Content Preview:\n{documents[0].page_content[:300]}...")
            
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please create a 'mynotes' folder in the same directory as this script")
        print("and add your .txt, .md, or .pdf files to it.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
