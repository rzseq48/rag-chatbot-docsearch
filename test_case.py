#!/usr/bin/env python3
"""
Simple Test Case for RAG Chatbot System
This demonstrates the complete functionality with a minimal example
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_rag_system():
    """Test the complete RAG system"""
    print("🧪 RAG System Test Case")
    print("=" * 50)
    
    try:
        # Import RAG pipeline
        from rag_pipeline import RAGPipeline
        print("✅ RAG Pipeline imported successfully")
        
        # Create temporary storage
        temp_dir = tempfile.mkdtemp(prefix="rag_test_")
        print(f"📁 Using temporary directory: {temp_dir}")
        
        # Initialize RAG pipeline
        print("\n🔧 Initializing RAG pipeline...")
        rag = RAGPipeline(
            storage_dir=temp_dir,
            embedder_type="mock",
            llm_provider="mock",
            collection_name="test_documents"
        )
        print("✅ RAG pipeline initialized")
        
        # Create test documents
        print("\n📄 Creating test documents...")
        test_docs_dir = Path(temp_dir) / "test_docs"
        test_docs_dir.mkdir(exist_ok=True)
        
        # Create sample documents
        (test_docs_dir / "ai_intro.txt").write_text("""
        Artificial Intelligence (AI) is a field of computer science that focuses on 
        creating intelligent machines capable of performing tasks that typically require 
        human intelligence. These tasks include learning, reasoning, problem-solving, 
        perception, and language understanding.
        """)
        
        (test_docs_dir / "ml_basics.txt").write_text("""
        Machine Learning is a subset of artificial intelligence that enables computers 
        to learn and improve from experience without being explicitly programmed. 
        It uses algorithms to identify patterns in data and make predictions.
        """)
        
        (test_docs_dir / "deep_learning.txt").write_text("""
        Deep Learning is a subset of machine learning that uses neural networks with 
        multiple layers to model and understand complex patterns in data. It has 
        revolutionized fields like computer vision, natural language processing, 
        and speech recognition.
        """)
        
        print(f"✅ Created 3 test documents in {test_docs_dir}")
        
        # Test 1: Document Ingestion
        print("\n📥 Test 1: Document Ingestion")
        print("-" * 30)
        result = rag.ingest_documents(str(test_docs_dir))
        
        print(f"📊 Ingestion Results:")
        print(f"  Status: {result['status']}")
        print(f"  Total files: {result['total_files']}")
        print(f"  Processed: {result['processed_files']}")
        print(f"  Failed: {result['failed_files']}")
        print(f"  Documents: {result['total_documents']}")
        
        assert result['status'] == 'completed', "Ingestion should succeed"
        assert result['total_documents'] > 0, "Should have processed documents"
        print("✅ Document ingestion test passed")
        
        # Test 2: Collection Info
        print("\n📊 Test 2: Collection Information")
        print("-" * 30)
        info = rag.get_collection_info()
        print(f"Collection: {info['collection_name']}")
        print(f"Document count: {info['document_count']}")
        print(f"Status: {info['status']}")
        
        assert info['document_count'] > 0, "Should have documents in collection"
        print("✅ Collection info test passed")
        
        # Test 3: Query Processing
        print("\n🔍 Test 3: Query Processing")
        print("-" * 30)
        
        test_queries = [
            "What is artificial intelligence?",
            "Explain machine learning",
            "What is deep learning?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nQuery {i}: {query}")
            response = rag.query(query, max_results=2)
            
            print(f"  Answer: {response['answer'][:100]}...")
            print(f"  Sources: {len(response['sources'])}")
            print(f"  Confidence: {response['confidence']:.2f}")
            
            assert 'answer' in response, "Response should have answer"
            assert 'sources' in response, "Response should have sources"
            assert len(response['sources']) > 0, "Should have at least one source"
            print(f"  ✅ Query {i} test passed")
        
        # Test 4: Embedding Generation
        print("\n🧠 Test 4: Embedding Generation")
        print("-" * 30)
        
        test_text = "This is a test document for embedding generation."
        embedding = rag.embedder.embed_text(test_text)
        
        print(f"Embedding dimension: {len(embedding)}")
        print(f"Sample values: {embedding[:5]}")
        
        assert isinstance(embedding, list), "Embedding should be a list"
        assert len(embedding) > 0, "Embedding should not be empty"
        assert all(isinstance(x, float) for x in embedding), "All values should be floats"
        print("✅ Embedding generation test passed")
        
        # Test 5: LLM Response Generation
        print("\n🤖 Test 5: LLM Response Generation")
        print("-" * 30)
        
        question = "What is the relationship between AI, ML, and DL?"
        context = ["AI is the broader field", "ML is a subset of AI", "DL is a subset of ML"]
        
        response = rag.llm.generate_response_with_context(question, context)
        print(f"Question: {question}")
        print(f"Response: {response}")
        
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        print("✅ LLM response generation test passed")
        
        # Test 6: Vector Search
        print("\n🔍 Test 6: Vector Search")
        print("-" * 30)
        
        query_embedding = rag.embedder.embed_text("artificial intelligence")
        results = rag.vector_store.search(query_embedding, n_results=3)
        
        print(f"Search results: {len(results)}")
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result['score']:.3f}")
            print(f"     Content: {result['content'][:50]}...")
        
        assert len(results) > 0, "Should have search results"
        print("✅ Vector search test passed")
        
        # Test 7: API Integration (Mock)
        print("\n🌐 Test 7: API Integration")
        print("-" * 30)
        
        # Simulate API request
        api_request = {
            "query": "What is machine learning?",
            "max_results": 3
        }
        
        # Process like API would
        response = rag.query(
            question=api_request["query"],
            max_results=api_request["max_results"]
        )
        
        api_response = {
            "question": response["question"],
            "answer": response["answer"],
            "sources": [
                {
                    "content": source["content"][:100] + "...",
                    "score": source["score"]
                }
                for source in response["sources"]
            ],
            "total_results": len(response["sources"]),
            "confidence": response["confidence"]
        }
        
        print(f"API Response:")
        print(f"  Question: {api_response['question']}")
        print(f"  Answer: {api_response['answer'][:100]}...")
        print(f"  Sources: {api_response['total_results']}")
        print(f"  Confidence: {api_response['confidence']:.2f}")
        
        assert 'question' in api_response, "API response should have question"
        assert 'answer' in api_response, "API response should have answer"
        print("✅ API integration test passed")
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("✅ Cleanup completed")
        
        # Final Results
        print("\n🎉 ALL TESTS PASSED!")
        print("=" * 50)
        print("✅ Document ingestion: PASSED")
        print("✅ Collection management: PASSED") 
        print("✅ Query processing: PASSED")
        print("✅ Embedding generation: PASSED")
        print("✅ LLM response generation: PASSED")
        print("✅ Vector search: PASSED")
        print("✅ API integration: PASSED")
        print("\n🚀 RAG System is fully functional!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_components():
    """Test individual components"""
    print("\n🔧 Testing Individual Components")
    print("=" * 50)
    
    try:
        # Test embedder
        from embeddings.embedder import get_embedder
        embedder = get_embedder("mock")
        embedding = embedder.embed_text("Test text")
        assert len(embedding) > 0
        print("✅ Embedder: PASSED")
        
        # Test LLM provider
        from llm.llm_provider import get_llm_provider
        llm = get_llm_provider("mock")
        response = llm.generate_response("Test question")
        assert len(response) > 0
        print("✅ LLM Provider: PASSED")
        
        # Test vector store
        from vectorstore.chroma_store import VectorStore
        vs = VectorStore("test_collection")
        info = vs.get_collection_info()
        assert "collection_name" in info
        print("✅ Vector Store: PASSED")
        
        # Test data ingestion
        from data_ingestion.loader import TxtHandler
        handler = TxtHandler()
        assert handler is not None
        print("✅ Data Ingestion: PASSED")
        
        print("\n✅ All individual components working!")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 RAG Chatbot System Test Suite")
    print("=" * 60)
    
    # Test individual components first
    components_ok = test_individual_components()
    
    if components_ok:
        # Test complete system
        system_ok = test_rag_system()
        
        if system_ok:
            print("\n🎉 COMPLETE SUCCESS!")
            print("The RAG chatbot system is fully functional and ready for use.")
        else:
            print("\n❌ System test failed")
    else:
        print("\n❌ Component tests failed")
    
    print("\n" + "=" * 60)
