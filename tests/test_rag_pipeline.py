"""
Comprehensive tests for RAG Pipeline
"""
import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from rag_pipeline import RAGPipeline
from data_ingestion.loader import Loader
from embeddings.embedder import get_embedder
from llm.llm_provider import get_llm_provider
from vectorstore.chroma_store import VectorStore

class TestRAGPipeline:
    """Test RAG Pipeline functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="rag_test_")
        self.rag = RAGPipeline(
            storage_dir=self.temp_dir,
            embedder_type="mock",
            llm_provider="mock",
            collection_name="test_documents"
        )
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_rag_initialization(self):
        """Test RAG pipeline initialization"""
        assert self.rag is not None
        assert self.rag.embedder is not None
        assert self.rag.llm is not None
        assert self.rag.vector_store is not None
    
    def test_embedder_functionality(self):
        """Test embedder functionality"""
        text = "This is a test document"
        embedding = self.rag.embedder.embed_text(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
    
    def test_llm_functionality(self):
        """Test LLM functionality"""
        question = "What is artificial intelligence?"
        response = self.rag.llm.generate_response(question)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_vector_store_functionality(self):
        """Test vector store functionality"""
        # Test adding documents
        documents = ["Test document 1", "Test document 2"]
        embeddings = [
            [0.1] * 384,  # Mock embeddings
            [0.2] * 384
        ]
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        
        success = self.rag.vector_store.add_documents(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        assert success is True
        
        # Test searching
        query_embedding = [0.15] * 384
        results = self.rag.vector_store.search(query_embedding, n_results=2)
        
        assert isinstance(results, list)
        assert len(results) <= 2
    
    def test_document_ingestion(self):
        """Test document ingestion"""
        # Create a test text file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("This is a test document for RAG pipeline testing.")
        
        # Ingest the document
        result = self.rag.ingest_documents(str(test_file))
        
        assert result["status"] == "completed"
        assert result["total_files"] >= 1
        assert result["processed_files"] >= 1
        assert result["total_documents"] >= 1
    
    def test_query_functionality(self):
        """Test query functionality"""
        # First ingest a document
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("Artificial intelligence is a field of computer science.")
        
        result = self.rag.ingest_documents(str(test_file))
        assert result["status"] == "completed"
        
        # Test query
        response = self.rag.query("What is artificial intelligence?")
        
        assert "question" in response
        assert "answer" in response
        assert "sources" in response
        assert response["question"] == "What is artificial intelligence?"
        assert isinstance(response["answer"], str)
        assert isinstance(response["sources"], list)
    
    def test_collection_info(self):
        """Test collection info retrieval"""
        info = self.rag.get_collection_info()
        
        assert "collection_name" in info
        assert "document_count" in info
        assert "status" in info
        assert info["collection_name"] == "test_documents"
    
    def test_clear_collection(self):
        """Test collection clearing"""
        # Add some documents first
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("Test content")
        
        self.rag.ingest_documents(str(test_file))
        
        # Clear collection
        success = self.rag.clear_collection()
        assert success is True
        
        # Verify collection is empty
        info = self.rag.get_collection_info()
        assert info["document_count"] == 0

class TestDataIngestion:
    """Test data ingestion components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="ingestion_test_")
        self.loader = Loader(cache_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_txt_handler(self):
        """Test text file handler"""
        from data_ingestion.loader import TxtHandler
        
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("This is a test text file.\nIt has multiple lines.")
        
        handler = TxtHandler()
        fd = {
            'path': test_file,
            'ext': '.txt',
            'source': str(test_file)
        }
        
        # Test inspection
        info = handler.inspect(fd)
        assert "page_count" in info
        assert "has_embedded_text" in info
        
        # Test extraction
        documents = list(handler.extract(fd))
        assert len(documents) > 0
        assert all('content' in doc for doc in documents)
    
    def test_pdf_handler(self):
        """Test PDF handler (if pypdf is available)"""
        try:
            from data_ingestion.loader import PdfHandler
            import pypdf
            
            # Create a simple PDF for testing
            # This would require creating an actual PDF file
            # For now, just test that the handler can be instantiated
            handler = PdfHandler()
            assert handler is not None
            
        except ImportError:
            pytest.skip("pypdf not available")
    
    def test_docx_handler(self):
        """Test DOCX handler (if python-docx is available)"""
        try:
            from data_ingestion.loader import DocxHandler
            from docx import Document
            
            # Create test DOCX file
            test_file = Path(self.temp_dir) / "test.docx"
            doc = Document()
            doc.add_paragraph("This is a test DOCX document.")
            doc.save(str(test_file))
            
            handler = DocxHandler()
            fd = {
                'path': test_file,
                'ext': '.docx',
                'source': str(test_file)
            }
            
            # Test inspection
            info = handler.inspect(fd)
            assert "page_count" in info
            
            # Test extraction
            documents = list(handler.extract(fd))
            assert len(documents) > 0
            
        except ImportError:
            pytest.skip("python-docx not available")

class TestEmbedders:
    """Test embedding functionality"""
    
    def test_mock_embedder(self):
        """Test mock embedder"""
        embedder = get_embedder("mock")
        
        # Test single text
        text = "Test document"
        embedding = embedder.embed_text(text)
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        
        # Test batch
        texts = ["Text 1", "Text 2"]
        embeddings = embedder.embed_batch(texts)
        assert len(embeddings) == 2
        assert all(isinstance(emb, list) for emb in embeddings)

class TestLLMProviders:
    """Test LLM providers"""
    
    def test_mock_llm(self):
        """Test mock LLM provider"""
        llm = get_llm_provider("mock")
        
        # Test basic response
        response = llm.generate_response("What is AI?")
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Test response with context
        context = ["AI is artificial intelligence"]
        response = llm.generate_response_with_context("What is AI?", context)
        assert isinstance(response, str)
        assert len(response) > 0

if __name__ == "__main__":
    pytest.main([__file__])
