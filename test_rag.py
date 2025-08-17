#!/usr/bin/env python3
"""
Test script for RAG MCP Server
"""

import json
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

def test_rag_status():
    """Test the rag_status tool"""
    try:
        from mcp_server.server import rag_status
        result = rag_status()
        print("✅ RAG Status Test:")
        print(json.dumps(result, indent=2))
        return True
    except Exception as e:
        print(f"❌ RAG Status Test Failed: {e}")
        return False

def test_rag_ingest():
    """Test the rag_ingest tool"""
    try:
        from mcp_server.server import rag_ingest
        
        # Test with a simple text
        test_data = {
            "texts": ["This is a test document for RAG testing."],
            "tags": ["test", "demo"]
        }
        
        result = rag_ingest(test_data)
        print("✅ RAG Ingest Test:")
        print(json.dumps(result, indent=2))
        return True
    except Exception as e:
        print(f"❌ RAG Ingest Test Failed: {e}")
        return False

def test_rag_query():
    """Test the rag_query tool"""
    try:
        from mcp_server.server import rag_query
        
        # Test with a simple query
        test_data = {
            "query": "What is this document about?",
            "k": 3
        }
        
        result = rag_query(test_data)
        print("✅ RAG Query Test:")
        print(json.dumps(result, indent=2))
        return True
    except Exception as e:
        print(f"❌ RAG Query Test Failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing RAG MCP Server...\n")
    
    tests = [
        test_rag_status,
        test_rag_ingest,
        test_rag_query
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! RAG MCP Server is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
