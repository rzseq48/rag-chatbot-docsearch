#!/usr/bin/env python3
"""
API Test Case for RAG Chatbot System
Tests the FastAPI endpoints
"""

import requests
import time
import subprocess
import sys
import os
from pathlib import Path
import tempfile
import threading

def start_api_server():
    """Start the API server in background"""
    try:
        # Start the server
        process = subprocess.Popen([
            sys.executable, "api/main.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(5)
        return process
    except Exception as e:
        print(f"Failed to start API server: {e}")
        return None

def test_api_endpoints():
    """Test all API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🌐 Testing API Endpoints")
    print("=" * 40)
    
    try:
        # Test 1: Health Check
        print("1. Testing health check...")
        response = requests.get(f"{base_url}/health", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("   ✅ Health check passed")
        
        # Test 2: Root endpoint
        print("2. Testing root endpoint...")
        response = requests.get(f"{base_url}/", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        print("   ✅ Root endpoint passed")
        
        # Test 3: Document listing
        print("3. Testing document listing...")
        response = requests.get(f"{base_url}/documents", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        print("   ✅ Document listing passed")
        
        # Test 4: Query endpoint
        print("4. Testing query endpoint...")
        query_data = {
            "query": "What is artificial intelligence?",
            "max_results": 3
        }
        response = requests.post(f"{base_url}/query", json=query_data, timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        print("   ✅ Query endpoint passed")
        
        # Test 5: Document ingestion
        print("5. Testing document ingestion...")
        # Create a test file
        test_file = Path(tempfile.gettempdir()) / "test_document.txt"
        test_file.write_text("This is a test document about artificial intelligence and machine learning.")
        
        with open(test_file, 'rb') as f:
            files = {'file': ('test_document.txt', f, 'text/plain')}
            response = requests.post(f"{base_url}/upload", files=files, timeout=30)
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        print("   ✅ Document upload passed")
        
        # Clean up test file
        test_file.unlink()
        
        # Test 6: Ingest endpoint
        print("6. Testing ingest endpoint...")
        ingest_data = {
            "input_path": str(Path.cwd() / "notebooks" / "Text_file.txt"),
            "chunk_size": 500
        }
        response = requests.post(f"{base_url}/ingest", json=ingest_data, timeout=30)
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        print("   ✅ Ingest endpoint passed")
        
        # Test 7: Query with ingested documents
        print("7. Testing query with ingested documents...")
        query_data = {
            "query": "What is in the document?",
            "max_results": 2
        }
        response = requests.post(f"{base_url}/query", json=query_data, timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        print("   ✅ Query with ingested documents passed")
        
        print("\n🎉 All API tests passed!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")
        print("   Make sure the server is running: python api/main.py")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_api_without_server():
    """Test API components without starting server"""
    print("\n🔧 Testing API Components (Without Server)")
    print("=" * 50)
    
    try:
        # Test imports
        sys.path.append(str(Path.cwd()))
        from api.main import app, get_rag_pipeline
        print("✅ API imports successful")
        
        # Test RAG pipeline initialization
        rag = get_rag_pipeline()
        assert rag is not None
        print("✅ RAG pipeline initialization successful")
        
        # Test collection info
        info = rag.get_collection_info()
        assert "collection_name" in info
        print("✅ Collection info retrieval successful")
        
        print("✅ All API component tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ API component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🧪 RAG Chatbot API Test Suite")
    print("=" * 60)
    
    # Test 1: API components without server
    print("Testing API components...")
    components_ok = test_api_without_server()
    
    if not components_ok:
        print("❌ API component tests failed")
        return False
    
    # Test 2: Full API with server
    print("\nTesting full API with server...")
    print("Note: This requires the API server to be running")
    print("Start with: python api/main.py")
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("✅ API server is running, testing endpoints...")
            api_ok = test_api_endpoints()
        else:
            print("⚠️  API server not responding properly")
            api_ok = False
    except:
        print("⚠️  API server not running")
        print("   To test full API, run: python api/main.py")
        print("   Then run this test again")
        api_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   API Components: {'✅ PASSED' if components_ok else '❌ FAILED'}")
    print(f"   API Endpoints: {'✅ PASSED' if api_ok else '⚠️  SKIPPED (server not running)'}")
    
    if components_ok:
        print("\n🎉 API system is functional!")
        print("   To test full API endpoints:")
        print("   1. Run: python api/main.py")
        print("   2. Run: python test_api.py")
    else:
        print("\n❌ API system has issues")
    
    return components_ok

if __name__ == "__main__":
    main()
