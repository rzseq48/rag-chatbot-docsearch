#!/usr/bin/env python3
"""
Quick Test Case for RAG System
Minimal test to verify everything works
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def quick_test():
    """Quick test of the RAG system"""
    print("⚡ Quick RAG System Test")
    print("=" * 40)
    
    try:
        # Test 1: Import all components
        print("1. Testing imports...")
        from rag_pipeline import RAGPipeline
        from embeddings.embedder import get_embedder
        from llm.llm_provider import get_llm_provider
        from vectorstore.chroma_store import VectorStore
        print("   ✅ All imports successful")
        
        # Test 2: Initialize components
        print("2. Testing component initialization...")
        embedder = get_embedder("mock")
        llm = get_llm_provider("mock")
        vs = VectorStore("quick_test")
        print("   ✅ All components initialized")
        
        # Test 3: Basic functionality
        print("3. Testing basic functionality...")
        
        # Test embedding
        embedding = embedder.embed_text("Test document")
        assert len(embedding) > 0
        print("   ✅ Embedding generation works")
        
        # Test LLM
        response = llm.generate_response("What is AI?")
        assert len(response) > 0
        print("   ✅ LLM response generation works")
        
        # Test vector store
        info = vs.get_collection_info()
        assert "collection_name" in info
        print("   ✅ Vector store works")
        
        # Test 4: RAG Pipeline
        print("4. Testing RAG pipeline...")
        rag = RAGPipeline(
            storage_dir="./quick_test_storage",
            embedder_type="mock",
            llm_provider="mock",
            collection_name="quick_test"
        )
        print("   ✅ RAG pipeline initialized")
        
        # Test 5: Document processing
        print("5. Testing document processing...")
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document about artificial intelligence.")
            temp_file = f.name
        
        result = rag.ingest_documents(temp_file)
        assert result['status'] == 'completed'
        print("   ✅ Document ingestion works")
        
        # Test 6: Query processing
        print("6. Testing query processing...")
        response = rag.query("What is artificial intelligence?")
        assert 'answer' in response
        assert 'sources' in response
        print("   ✅ Query processing works")
        
        # Cleanup
        import os
        import shutil
        os.unlink(temp_file)
        shutil.rmtree("./quick_test_storage", ignore_errors=True)
        print("   ✅ Cleanup completed")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("The RAG system is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🚀 Ready to use the RAG system!")
        print("Run 'python demo.py' for a full demonstration")
        print("Run 'python api/main.py' to start the API server")
    else:
        print("\n❌ System has issues - check the error messages above")
    
    sys.exit(0 if success else 1)
