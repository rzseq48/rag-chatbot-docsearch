#!/usr/bin/env python3
"""
Complete RAG System Example

This example demonstrates the full RAG pipeline:
1. Document ingestion
2. Vector storage
3. Query processing
4. Response generation
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag_pipeline import RAGPipeline

def main():
    """Complete RAG system demonstration"""
    print("🚀 RAG Chatbot Complete Example")
    print("=" * 50)
    
    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    rag = RAGPipeline(
        storage_dir="./storage",
        embedder_type="mock",  # Use mock for demo (no API keys needed)
        llm_provider="mock",   # Use mock for demo
        collection_name="demo_documents"
    )
    print("✅ RAG pipeline initialized")
    
    # Create sample documents
    print("\n📄 Creating sample documents...")
    sample_docs_dir = Path("./sample_docs")
    sample_docs_dir.mkdir(exist_ok=True)
    
    # Create sample text files
    (sample_docs_dir / "ai_basics.txt").write_text("""
    Artificial Intelligence (AI) is a branch of computer science that aims to create 
    intelligent machines that can perform tasks that typically require human intelligence.
    These tasks include learning, reasoning, problem-solving, perception, and language understanding.
    """)
    
    (sample_docs_dir / "machine_learning.txt").write_text("""
    Machine Learning is a subset of artificial intelligence that focuses on algorithms 
    that can learn and make decisions from data. It includes supervised learning, 
    unsupervised learning, and reinforcement learning techniques.
    """)
    
    (sample_docs_dir / "deep_learning.txt").write_text("""
    Deep Learning is a subset of machine learning that uses neural networks with multiple 
    layers to model and understand complex patterns in data. It has revolutionized fields 
    like computer vision, natural language processing, and speech recognition.
    """)
    
    print(f"✅ Created sample documents in {sample_docs_dir}")
    
    # Ingest documents
    print("\n📥 Ingesting documents...")
    result = rag.ingest_documents(str(sample_docs_dir))
    
    print(f"📊 Ingestion Results:")
    print(f"  - Total files: {result['total_files']}")
    print(f"  - Processed files: {result['processed_files']}")
    print(f"  - Failed files: {result['failed_files']}")
    print(f"  - Total documents: {result['total_documents']}")
    print(f"  - Status: {result['status']}")
    
    # Get collection info
    print("\n📊 Collection Information:")
    info = rag.get_collection_info()
    print(f"  - Collection: {info['collection_name']}")
    print(f"  - Document count: {info['document_count']}")
    print(f"  - Status: {info['status']}")
    
    # Test queries
    print("\n🔍 Testing Queries:")
    queries = [
        "What is artificial intelligence?",
        "Explain machine learning",
        "What is deep learning?",
        "How are AI, ML, and DL related?",
        "What are the applications of AI?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: {query}")
        response = rag.query(query, max_results=3)
        
        print(f"   Answer: {response['answer']}")
        print(f"   Sources: {len(response['sources'])}")
        print(f"   Confidence: {response['confidence']:.2f}")
        
        if response['sources']:
            print("   Top sources:")
            for j, source in enumerate(response['sources'][:2], 1):
                content_preview = source['content'][:100] + "..." if len(source['content']) > 100 else source['content']
                print(f"     {j}. {content_preview}")
    
    # Demonstrate advanced features
    print("\n🔧 Advanced Features:")
    
    # Test with different chunk sizes
    print("Testing different chunk sizes...")
    small_rag = RAGPipeline(
        storage_dir="./storage_small",
        embedder_type="mock",
        llm_provider="mock",
        collection_name="small_chunks"
    )
    
    # Clear and test with small chunks
    small_rag.clear_collection()
    result = small_rag.ingest_documents(str(sample_docs_dir), chunk_size=200, chunk_overlap=50)
    print(f"Small chunks result: {result['total_documents']} documents")
    
    # Test query with small chunks
    response = small_rag.query("What is AI?", max_results=5)
    print(f"Small chunks query: {len(response['sources'])} sources found")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    import shutil
    shutil.rmtree(sample_docs_dir, ignore_errors=True)
    shutil.rmtree("./storage_small", ignore_errors=True)
    print("✅ Cleanup completed")
    
    print("\n🎉 RAG System Demo Completed Successfully!")
    print("\nNext steps:")
    print("1. Run the API server: python api/main.py")
    print("2. Test with real documents")
    print("3. Configure with real LLM providers (OpenAI, Anthropic)")
    print("4. Deploy with Docker: docker-compose up")

if __name__ == "__main__":
    main()
