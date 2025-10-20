#!/usr/bin/env python3
"""
RAG Chatbot System Demo
A simple demonstration of the complete system
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def demo_rag_system():
    """Demonstrate the RAG system"""
    print("🎯 RAG Chatbot System Demo")
    print("=" * 50)
    
    try:
        # Import and initialize
        from rag_pipeline import RAGPipeline
        print("✅ Imported RAG Pipeline")
        
        # Initialize with mock providers (no API keys needed)
        rag = RAGPipeline(
            storage_dir="./demo_storage",
            embedder_type="mock",
            llm_provider="mock",
            collection_name="demo_docs"
        )
        print("✅ Initialized RAG Pipeline")
        
        # Create demo documents
        print("\n📄 Creating demo documents...")
        demo_dir = Path("./demo_documents")
        demo_dir.mkdir(exist_ok=True)
        
        # Sample documents about AI
        documents = {
            "ai_overview.txt": """
            Artificial Intelligence (AI) is a branch of computer science that aims to create 
            intelligent machines capable of performing tasks that typically require human intelligence. 
            AI includes machine learning, natural language processing, computer vision, and robotics.
            """,
            
            "machine_learning.txt": """
            Machine Learning is a subset of artificial intelligence that enables computers to learn 
            and improve from experience without being explicitly programmed. It uses algorithms to 
            identify patterns in data and make predictions or decisions.
            """,
            
            "deep_learning.txt": """
            Deep Learning is a subset of machine learning that uses neural networks with multiple 
            layers to model and understand complex patterns in data. It has revolutionized fields 
            like computer vision, natural language processing, and speech recognition.
            """,
            
            "neural_networks.txt": """
            Neural Networks are computing systems inspired by biological neural networks. They consist 
            of interconnected nodes (neurons) that process information and can learn to perform 
            specific tasks through training on data.
            """
        }
        
        for filename, content in documents.items():
            (demo_dir / filename).write_text(content.strip())
        
        print(f"✅ Created {len(documents)} demo documents")
        
        # Ingest documents
        print("\n📥 Ingesting documents...")
        result = rag.ingest_documents(str(demo_dir))
        print(f"✅ Processed {result['total_documents']} document chunks")
        
        # Demo queries
        print("\n🔍 Demo Queries:")
        print("-" * 30)
        
        queries = [
            "What is artificial intelligence?",
            "Explain machine learning",
            "What is deep learning?",
            "How do neural networks work?",
            "What are the applications of AI?"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n{i}. Query: {query}")
            response = rag.query(query, max_results=2)
            
            print(f"   Answer: {response['answer']}")
            print(f"   Sources: {len(response['sources'])}")
            print(f"   Confidence: {response['confidence']:.2f}")
            
            if response['sources']:
                print("   Top source:")
                source = response['sources'][0]
                preview = source['content'][:100] + "..." if len(source['content']) > 100 else source['content']
                print(f"   \"{preview}\"")
        
        # Show system capabilities
        print("\n🔧 System Capabilities:")
        print("-" * 30)
        
        # Test embedding
        embedding = rag.embedder.embed_text("Test document")
        print(f"✅ Embedding generation: {len(embedding)} dimensions")
        
        # Test LLM
        llm_response = rag.llm.generate_response("What is AI?")
        print(f"✅ LLM response generation: {len(llm_response)} characters")
        
        # Test vector search
        search_results = rag.vector_store.search(embedding, n_results=3)
        print(f"✅ Vector search: {len(search_results)} results")
        
        # Collection info
        info = rag.get_collection_info()
        print(f"✅ Collection: {info['document_count']} documents stored")
        
        # Cleanup
        print("\n🧹 Cleaning up demo files...")
        import shutil
        shutil.rmtree(demo_dir, ignore_errors=True)
        shutil.rmtree("./demo_storage", ignore_errors=True)
        print("✅ Cleanup completed")
        
        print("\n🎉 Demo completed successfully!")
        print("\n🚀 The RAG system is fully functional!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_api_usage():
    """Show how to use the API"""
    print("\n🌐 API Usage Examples:")
    print("=" * 50)
    
    print("1. Start the API server:")
    print("   python api/main.py")
    print()
    
    print("2. Test the API endpoints:")
    print("   # Health check")
    print("   curl http://localhost:8000/health")
    print()
    print("   # Upload a document")
    print("   curl -X POST http://localhost:8000/upload \\")
    print("     -F 'file=@document.pdf'")
    print()
    print("   # Query documents")
    print("   curl -X POST http://localhost:8000/query \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"query\": \"What is AI?\", \"max_results\": 5}'")
    print()
    print("   # Ingest from directory")
    print("   curl -X POST http://localhost:8000/ingest \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"input_path\": \"/path/to/documents\"}'")

def show_docker_usage():
    """Show Docker usage"""
    print("\n🐳 Docker Usage:")
    print("=" * 50)
    
    print("1. Build and run with Docker Compose:")
    print("   docker-compose up")
    print()
    print("2. Or build and run manually:")
    print("   docker build -t rag-chatbot .")
    print("   docker run -p 8000:8000 rag-chatbot")
    print()
    print("3. With volume mounts for persistent storage:")
    print("   docker run -p 8000:8000 \\")
    print("     -v $(pwd)/storage:/app/storage \\")
    print("     -v $(pwd)/data:/app/data \\")
    print("     rag-chatbot")

def main():
    """Main demo function"""
    print("🎯 RAG Chatbot System - Complete Demo")
    print("=" * 80)
    
    # Run the demo
    success = demo_rag_system()
    
    if success:
        # Show usage examples
        show_api_usage()
        show_docker_usage()
        
        print("\n" + "=" * 80)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("The RAG chatbot system is fully functional and ready for use.")
        print("\nNext steps:")
        print("1. Run tests: python run_tests.py")
        print("2. Start API: python api/main.py")
        print("3. Deploy: docker-compose up")
    else:
        print("\n❌ Demo failed - check the error messages above")

if __name__ == "__main__":
    main()
