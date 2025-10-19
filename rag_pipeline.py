"""
Complete RAG Pipeline Implementation
Connects all components: data ingestion, embeddings, vector store, and LLM
"""
import os
import logging
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path

# Import all components
from data_ingestion.loader import Loader, OcrClient, StatusStore
from embeddings.embedder import get_embedder
from llm.llm_provider import get_llm_provider
from vectorstore.chroma_store import VectorStore
from storage.db_config import DBConfig

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Complete RAG Pipeline that orchestrates all components
    """
    
    def __init__(
        self,
        storage_dir: str = "./storage",
        embedder_type: str = "mock",
        llm_provider: str = "mock",
        collection_name: str = "documents",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize RAG Pipeline
        
        Args:
            storage_dir: Directory for ChromaDB storage
            embedder_type: Type of embedder ("mock", "openai", "sentence_bert")
            llm_provider: Type of LLM provider ("mock", "openai", "anthropic")
            collection_name: ChromaDB collection name
            cache_dir: Cache directory for temporary files
        """
        self.storage_dir = storage_dir
        self.collection_name = collection_name
        
        # Set up environment variables
        os.environ["STORAGE_DIR"] = storage_dir
        os.environ["VECTOR_BACKEND"] = "chroma"
        
        # Initialize components
        self._setup_components(embedder_type, llm_provider, cache_dir)
        
        logger.info("RAG Pipeline initialized successfully")
    
    def _setup_components(self, embedder_type: str, llm_provider: str, cache_dir: Optional[str]):
        """Setup all pipeline components"""
        
        # Initialize embedder
        self.embedder = get_embedder(embedder_type)
        logger.info(f"Initialized embedder: {embedder_type}")
        
        # Initialize LLM provider
        self.llm = get_llm_provider(llm_provider)
        logger.info(f"Initialized LLM provider: {llm_provider}")
        
        # Initialize vector store
        self.vector_store = VectorStore(collection_name=self.collection_name)
        logger.info(f"Initialized vector store: {self.collection_name}")
        
        # Initialize OCR client
        self.ocr_client = OcrClient(provider="local")
        
        # Initialize status store
        self.status_store = StatusStore(cache_dir=cache_dir)
        
        # Initialize data loader
        self.loader = Loader(
            cache_dir=cache_dir,
            ocr_client=self.ocr_client,
            status_store=self.status_store,
            vector_client=self.vector_store.adapter,
            index_on_ingest=True
        )
        logger.info("Initialized data loader")
    
    def ingest_documents(
        self, 
        input_path: str, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> Dict[str, Any]:
        """
        Ingest documents into the RAG system
        
        Args:
            input_path: Path to document(s) or directory
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
        
        Returns:
            Dictionary with ingestion results
        """
        logger.info(f"Starting document ingestion from: {input_path}")
        
        try:
            # Discover files
            file_descriptors = list(self.loader.discover(input_path))
            logger.info(f"Discovered {len(file_descriptors)} files")
            
            total_documents = 0
            processed_files = 0
            failed_files = 0
            
            # Process each file
            for fd in file_descriptors:
                try:
                    logger.info(f"Processing file: {fd.get('source', 'unknown')}")
                    
                    # Extract documents from file
                    documents = list(self.loader.process_file(fd))
                    
                    if documents:
                        # Chunk documents if needed
                        chunked_docs = self._chunk_documents(documents, chunk_size, chunk_overlap)
                        
                        # Generate embeddings
                        embeddings = self._generate_embeddings(chunked_docs)
                        
                        # Store in vector database
                        self._store_documents(chunked_docs, embeddings)
                        
                        total_documents += len(chunked_docs)
                        processed_files += 1
                        
                        logger.info(f"Processed {len(chunked_docs)} chunks from {fd.get('source')}")
                    else:
                        logger.warning(f"No documents extracted from {fd.get('source')}")
                        failed_files += 1
                        
                except Exception as e:
                    logger.error(f"Error processing file {fd.get('source')}: {e}")
                    failed_files += 1
                    continue
            
            result = {
                "status": "completed",
                "total_files": len(file_descriptors),
                "processed_files": processed_files,
                "failed_files": failed_files,
                "total_documents": total_documents,
                "collection_name": self.collection_name
            }
            
            logger.info(f"Ingestion completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "total_files": 0,
                "processed_files": 0,
                "failed_files": 0,
                "total_documents": 0
            }
    
    def _chunk_documents(self, documents: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
        """Chunk documents into smaller pieces"""
        chunked_docs = []
        
        for doc in documents:
            content = doc.get('content', '')
            if len(content) <= chunk_size:
                chunked_docs.append(doc)
            else:
                # Split into chunks
                chunks = self._split_text(content, chunk_size, chunk_overlap)
                for i, chunk in enumerate(chunks):
                    chunk_doc = doc.copy()
                    chunk_doc['content'] = chunk
                    chunk_doc['chunk_id'] = f"{doc.get('source', 'doc')}_chunk_{i}"
                    chunk_doc['chunk_index'] = i
                    chunk_doc['total_chunks'] = len(chunks)
                    chunked_docs.append(chunk_doc)
        
        return chunked_docs
    
    def _split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= len(text):
                break
                
            start = end - chunk_overlap
        
        return chunks
    
    def _generate_embeddings(self, documents: List[Dict[str, Any]]) -> List[List[float]]:
        """Generate embeddings for documents"""
        texts = [doc['content'] for doc in documents]
        return self.embedder.embed_batch(texts)
    
    def _store_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Store documents in vector database"""
        ids = [doc.get('chunk_id', f"doc_{i}") for i, doc in enumerate(documents)]
        metadatas = [doc.get('metadata', {}) for doc in documents]
        
        self.vector_store.add_documents(
            documents=[doc['content'] for doc in documents],
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(
        self, 
        question: str, 
        max_results: int = 5,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User question
            max_results: Maximum number of results to return
            include_metadata: Whether to include metadata in results
        
        Returns:
            Dictionary with query results and answer
        """
        logger.info(f"Processing query: {question}")
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_text(question)
            
            # Search for similar documents
            results = self.vector_store.search(
                query_embedding=query_embedding,
                n_results=max_results
            )
            
            if not results:
                return {
                    "question": question,
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Extract context from results
            context_documents = [result['content'] for result in results]
            
            # Generate answer using LLM
            answer = self.llm.generate_response_with_context(question, context_documents)
            
            # Prepare sources
            sources = []
            for result in results:
                source = {
                    "content": result['content'][:200] + "..." if len(result['content']) > 200 else result['content'],
                    "score": result['score'],
                    "metadata": result['metadata'] if include_metadata else {}
                }
                sources.append(source)
            
            response = {
                "question": question,
                "answer": answer,
                "sources": sources,
                "confidence": max([result['score'] for result in results]) if results else 0.0,
                "total_sources": len(results)
            }
            
            logger.info(f"Query completed with {len(results)} sources")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "question": question,
                "answer": f"Sorry, I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the document collection"""
        return self.vector_store.get_collection_info()
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        return self.vector_store.clear_collection()

# Example usage
if __name__ == "__main__":
    # Initialize RAG pipeline
    rag = RAGPipeline(
        storage_dir="./storage",
        embedder_type="mock",
        llm_provider="mock"
    )
    
    # Ingest documents
    result = rag.ingest_documents("./notebooks/Text_file.txt")
    print(f"Ingestion result: {result}")
    
    # Query the system
    response = rag.query("What is this document about?")
    print(f"Query response: {response}")
    
    # Get collection info
    info = rag.get_collection_info()
    print(f"Collection info: {info}")
