#!/usr/bin/env python3
"""
Test Runner for RAG Chatbot System
Runs all tests and provides a comprehensive report
"""

import sys
import subprocess
import time
from pathlib import Path

def run_test(test_file, description):
    """Run a test file and return results"""
    print(f"\n🧪 Running {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run([
            sys.executable, test_file
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(f"✅ {description}: PASSED")
            return True
        else:
            print(f"❌ {description}: FAILED")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description}: TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {description}: ERROR - {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 RAG Chatbot System - Complete Test Suite")
    print("=" * 80)
    
    # List of tests to run
    tests = [
        ("test_case.py", "Core RAG System Test"),
        ("test_api.py", "API System Test"),
        ("tests/test_chroma_adapter_smoke.py", "ChromaDB Smoke Test"),
    ]
    
    results = {}
    
    # Run each test
    for test_file, description in tests:
        if Path(test_file).exists():
            results[description] = run_test(test_file, description)
        else:
            print(f"⚠️  {description}: SKIPPED (file not found: {test_file})")
            results[description] = None
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for description, result in results.items():
        if result is True:
            print(f"✅ {description}: PASSED")
            passed += 1
        elif result is False:
            print(f"❌ {description}: FAILED")
            failed += 1
        else:
            print(f"⚠️  {description}: SKIPPED")
            skipped += 1
    
    print("\n" + "=" * 80)
    print(f"📈 Total Results: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("The RAG chatbot system is fully functional!")
        
        print("\n🚀 Next Steps:")
        print("1. Start the API server: python api/main.py")
        print("2. Test with real documents")
        print("3. Configure with real LLM providers")
        print("4. Deploy with Docker: docker-compose up")
        
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        print("Check the output above for details")
    
    print("\n" + "=" * 80)
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
