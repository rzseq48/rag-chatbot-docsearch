#!/usr/bin/env python3
"""
Groq LLM Example
Demonstrates how to use Groq for fast inference in the RAG system
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def groq_example():
    """Example using Groq LLM provider"""
    print("🚀 Groq LLM Example")
    print("=" * 50)
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not found")
        print("Please set your Groq API key:")
        print("export GROQ_API_KEY=your_groq_api_key_here")
        return False
    
    try:
        # Initialize RAG pipeline with Groq
        from rag_pipeline import RAGPipeline
        
        print("🔧 Initializing RAG pipeline with Groq...")
        rag = RAGPipeline(
            storage_dir="./groq_example_storage",
            embedder_type="mock",  # Use mock for demo
            llm_provider="groq",   # Use Groq for fast inference
            collection_name="groq_demo"
        )
        print("✅ RAG pipeline with Groq initialized")
        
        # Create sample documents
        print("\n📄 Creating sample documents...")
        sample_docs = {
            "ai_overview.txt": """
            Artificial Intelligence (AI) is revolutionizing industries across the globe. 
            From healthcare to finance, AI technologies are enabling unprecedented levels 
            of automation, efficiency, and innovation. Machine learning algorithms can 
            process vast amounts of data to identify patterns and make predictions that 
            were previously impossible.
            """,
            
            "ml_applications.txt": """
            Machine Learning applications are everywhere in modern technology. 
            Recommendation systems power e-commerce platforms, computer vision 
            enables autonomous vehicles, and natural language processing drives 
            chatbots and virtual assistants. These applications are transforming 
            how we interact with technology and solve complex problems.
            """,
            
            "future_ai.txt": """
            The future of AI holds incredible promise. We're moving toward artificial 
            general intelligence (AGI) that could match or exceed human cognitive 
            abilities. Quantum computing, neuromorphic chips, and advanced neural 
            architectures are pushing the boundaries of what's possible with AI.
            """
        }
        
        # Create temporary directory for documents
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="groq_demo_")
        
        for filename, content in sample_docs.items():
            (Path(temp_dir) / filename).write_text(content.strip())
        
        print(f"✅ Created {len(sample_docs)} sample documents")
        
        # Ingest documents
        print("\n📥 Ingesting documents...")
        result = rag.ingest_documents(temp_dir)
        print(f"✅ Processed {result['total_documents']} document chunks")
        
        # Test queries with Groq
        print("\n🔍 Testing queries with Groq...")
        queries = [
            "How is AI revolutionizing industries?",
            "What are the main applications of machine learning?",
            "What does the future of AI look like?",
            "How do recommendation systems work?",
            "What is artificial general intelligence?"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n{i}. Query: {query}")
            response = rag.query(query, max_results=2)
            
            print(f"   Answer: {response['answer']}")
            print(f"   Sources: {len(response['sources'])}")
            print(f"   Confidence: {response['confidence']:.2f}")
        
        # Test different Groq models
        print("\n🤖 Testing different Groq models...")
        models_to_test = [
            "llama3-8b-8192",
            "llama3-70b-8192", 
            "mixtral-8x7b-32768"
        ]
        
        for model in models_to_test:
            try:
                print(f"\nTesting model: {model}")
                from llm.llm_provider import get_llm_provider
                
                llm = get_llm_provider("groq", model=model)
                response = llm.generate_response("What is artificial intelligence?")
                print(f"   Response: {response[:100]}...")
                print(f"   ✅ Model {model} working")
                
            except Exception as e:
                print(f"   ❌ Model {model} failed: {e}")
        
        # Performance comparison
        print("\n⚡ Performance comparison...")
        import time
        
        test_prompt = "Explain the benefits of artificial intelligence in healthcare."
        
        # Test Groq speed
        start_time = time.time()
        groq_response = rag.llm.generate_response(test_prompt)
        groq_time = time.time() - start_time
        
        print(f"Groq response time: {groq_time:.2f} seconds")
        print(f"Groq response: {groq_response[:100]}...")
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree("./groq_example_storage", ignore_errors=True)
        print("✅ Cleanup completed")
        
        print("\n🎉 Groq example completed successfully!")
        print("\n🚀 Groq provides:")
        print("   - Ultra-fast inference (10-100x faster than traditional APIs)")
        print("   - Multiple model options (Llama, Mixtral, Gemma)")
        print("   - Cost-effective for high-volume usage")
        print("   - Excellent for real-time applications")
        
        return True
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_groq_benefits():
    """Show benefits of using Groq"""
    print("\n💡 Why Use Groq?")
    print("=" * 50)
    
    print("🚀 Speed Benefits:")
    print("   - 10-100x faster than traditional cloud APIs")
    print("   - Sub-second response times for most queries")
    print("   - Perfect for real-time applications")
    print()
    
    print("💰 Cost Benefits:")
    print("   - Free tier with generous limits")
    print("   - Pay-per-use pricing model")
    print("   - No minimum commitments")
    print()
    
    print("🔧 Technical Benefits:")
    print("   - Multiple model options")
    print("   - Simple API integration")
    print("   - Reliable uptime")
    print("   - Global edge locations")
    print()
    
    print("📊 Available Models:")
    print("   - llama3-8b-8192: Fast, efficient, good for most tasks")
    print("   - llama3-70b-8192: More capable, better reasoning")
    print("   - mixtral-8x7b-32768: Excellent for complex reasoning")
    print("   - gemma2-9b-it: Google's efficient model")
    print("   - gemma2-27b-it: Larger Google model")

def main():
    """Main function"""
    print("🧪 Groq LLM Integration Example")
    print("=" * 60)
    
    # Show benefits
    show_groq_benefits()
    
    # Run example
    success = groq_example()
    
    if success:
        print("\n🎉 Groq integration is working perfectly!")
        print("\nTo use Groq in your RAG system:")
        print("1. Set GROQ_API_KEY environment variable")
        print("2. Set LLM_PROVIDER=groq in .env file")
        print("3. Start your API: python api/main.py")
    else:
        print("\n❌ Groq integration needs setup")
        print("1. Get API key from: https://console.groq.com/")
        print("2. Install: pip install groq")
        print("3. Set: export GROQ_API_KEY=your_key")

if __name__ == "__main__":
    main()
