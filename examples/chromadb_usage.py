#!/usr/bin/env python3
"""
ChromaDB Usage Example for RAG Chatbot

This example demonstrates how to:
1. Set up ChromaDB with proper configuration
2. Add documents to the vector store
3. Query the vector store for similar documents
4. Use the existing ChromaAdapter class
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from storage.db_config import DBConfig
from storage.chroma_adapter import ChromaAdapter

def setup_chromadb_example():
    """Example of setting up ChromaDB with environment variables"""
    
    # Set up environment variables for ChromaDB
    temp_dir = tempfile.mkdtemp(prefix="chromadb_example_")
    print(f"Using temporary directory: {temp_dir}")
    
    # Configure environment
    os.environ["STORAGE_DIR"] = temp_dir
    os.environ["VECTOR_BACKEND"] = "chroma"
    os.environ["SQLITE_FILENAME"] = "example_meta.sqlite"
    
    try:
        # Create configuration
        config = DBConfig.from_env()
        print(f"Storage directory: {config.base_dir}")
        print(f"Vector backend: {config.vector_backend}")
        
        # Initialize ChromaDB adapter
        adapter = ChromaAdapter(config, collection_name="example_documents")
        print("✅ ChromaDB adapter initialized successfully!")
        
        return adapter, temp_dir
        
    except ImportError as e:
        print(f"❌ ChromaDB not installed: {e}")
        print("Install with: pip install chromadb")
        return None, temp_dir
    except Exception as e:
        print(f"❌ Error setting up ChromaDB: {e}")
        return None, temp_dir

def add_sample_documents(adapter):
    """Add sample documents to ChromaDB"""
    
    # Sample documents
    documents = [
        "Artificial intelligence is transforming healthcare with diagnostic tools.",
        "Machine learning algorithms can analyze medical images for early disease detection.",
        "Natural language processing helps doctors extract insights from patient records.",
        "Computer vision systems assist in surgical procedures and patient monitoring.",
        "AI-powered chatbots provide 24/7 patient support and symptom checking."
    ]
    
    # Generate simple embeddings (in real usage, use proper embedding models)
    import random
    embedding_dim = 384  # Common embedding dimension
    
    ids = [f"doc_{i}" for i in range(len(documents))]
    embeddings = [[random.random() for _ in range(embedding_dim)] for _ in documents]
    metadatas = [{"source": "example", "topic": "healthcare_ai"} for _ in documents]
    
    print("Adding documents to ChromaDB...")
    adapter.add_documents(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )
    print(f"✅ Added {len(documents)} documents successfully!")
    
    return embeddings[0]  # Return first embedding for querying

def query_documents(adapter, query_embedding):
    """Query ChromaDB for similar documents"""
    results = adapter.query(query_embedding=query_embedding, n_results=3)
    
    print("Query results:")
    if results and 'documents' in results:
        for i, doc in enumerate(results['documents'][0]):
            print(f"  {i+1}. {doc}")
    
    return results

def cleanup(temp_dir):
    """Clean up temporary files"""
    try:
        shutil.rmtree(temp_dir)
        print(f"✅ Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"⚠️  Warning: Could not clean up {temp_dir}: {e}")

def main():
    """Main example function"""
    print("🚀 ChromaDB Setup and Usage Example")
    print("=" * 50)
    
    # Setup ChromaDB
    adapter, temp_dir = setup_chromadb_example()
    
    if adapter is None:
        print("❌ Failed to setup ChromaDB. Please check installation.")
        return
    
    try:
        # Add documents
        query_embedding = add_sample_documents(adapter)
        
        # Query documents
        results = query_documents(adapter, query_embedding)
        
        print("\n✅ ChromaDB example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during example: {e}")
    
    finally:
        # Cleanup
        cleanup(temp_dir)

if __name__ == "__main__":
    main()
