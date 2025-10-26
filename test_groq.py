#!/usr/bin/env python3
"""
Test Groq LLM Integration
Tests the Groq LLM provider with your API key
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_groq_llm():
    """Test Groq LLM provider"""
    print("🚀 Testing Groq LLM Integration")
    print("=" * 50)
    
    try:
        # Check if GROQ_API_KEY is set
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("❌ GROQ_API_KEY environment variable not set")
            print("Please set your Groq API key:")
            print("export GROQ_API_KEY=your_groq_api_key_here")
            return False
        
        print(f"✅ GROQ_API_KEY found: {api_key[:10]}...")
        
        # Test Groq LLM provider
        from llm.llm_provider import get_llm_provider
        
        print("\n🔧 Initializing Groq LLM provider...")
        llm = get_llm_provider("groq", model="llama3-8b-8192")
        print("✅ Groq LLM provider initialized")
        
        # Test basic response
        print("\n💬 Testing basic response...")
        response = llm.generate_response("What is artificial intelligence?")
        print(f"Response: {response}")
        print("✅ Basic response test passed")
        
        # Test response with context
        print("\n📚 Testing response with context...")
        context_docs = [
            "Artificial Intelligence (AI) is a field of computer science.",
            "Machine Learning is a subset of AI that enables computers to learn.",
            "Deep Learning uses neural networks to solve complex problems."
        ]
        
        response_with_context = llm.generate_response_with_context(
            "What is the relationship between AI, ML, and DL?",
            context_docs
        )
        print(f"Response with context: {response_with_context}")
        print("✅ Context response test passed")
        
        # Test RAG pipeline with Groq
        print("\n🔗 Testing RAG pipeline with Groq...")
        from rag_pipeline import RAGPipeline
        
        rag = RAGPipeline(
            storage_dir="./groq_test_storage",
            embedder_type="mock",
            llm_provider="groq",
            collection_name="groq_test"
        )
        print("✅ RAG pipeline with Groq initialized")
        
        # Create test document
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Artificial Intelligence is transforming industries worldwide. Machine learning algorithms can analyze vast amounts of data to identify patterns and make predictions.")
            temp_file = f.name
        
        # Ingest document
        result = rag.ingest_documents(temp_file)
        print(f"✅ Document ingested: {result['total_documents']} chunks")
        
        # Query with Groq
        response = rag.query("How is AI transforming industries?")
        print(f"RAG Response: {response['answer']}")
        print(f"Sources: {len(response['sources'])}")
        print("✅ RAG pipeline with Groq test passed")
        
        # Cleanup
        import shutil
        os.unlink(temp_file)
        shutil.rmtree("./groq_test_storage", ignore_errors=True)
        print("✅ Cleanup completed")
        
        print("\n🎉 All Groq tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Install Groq package: pip install groq")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_groq_setup():
    """Show how to set up Groq"""
    print("\n🔧 Groq Setup Instructions:")
    print("=" * 50)
    
    print("1. Get your Groq API key:")
    print("   - Visit: https://console.groq.com/")
    print("   - Sign up for a free account")
    print("   - Get your API key from the dashboard")
    print()
    
    print("2. Set your API key:")
    print("   export GROQ_API_KEY=your_api_key_here")
    print("   # Or add to .env file:")
    print("   echo 'GROQ_API_KEY=your_api_key_here' >> .env")
    print()
    
    print("3. Install Groq package:")
    print("   pip install groq")
    print()
    
    print("4. Configure your system:")
    print("   # In .env file:")
    print("   LLM_PROVIDER=groq")
    print("   GROQ_API_KEY=your_api_key_here")
    print()
    
    print("5. Available Groq models:")
    print("   - llama3-8b-8192 (default)")
    print("   - llama3-70b-8192")
    print("   - mixtral-8x7b-32768")
    print("   - gemma2-9b-it")
    print("   - gemma2-27b-it")

def main():
    """Main test function"""
    print("🧪 Groq LLM Integration Test")
    print("=" * 60)
    
    # Check if Groq is available
    try:
        import groq
        print("✅ Groq package is installed")
    except ImportError:
        print("❌ Groq package not installed")
        print("Install with: pip install groq")
        show_groq_setup()
        return False
    
    # Run the test
    success = test_groq_llm()
    
    if success:
        print("\n🎉 Groq integration is working!")
        print("\nTo use Groq in your RAG system:")
        print("1. Set GROQ_API_KEY environment variable")
        print("2. Set LLM_PROVIDER=groq in your .env file")
        print("3. Run: python api/main.py")
    else:
        print("\n❌ Groq integration failed")
        show_groq_setup()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
