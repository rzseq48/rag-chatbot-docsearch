"""
Vector store implementation using ChromaDB
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from storage.chroma_adapter import ChromaAdapter
from storage.db_config import DBConfig
import os

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector store wrapper for ChromaDB operations
    """
    
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.adapter = None
        self._initialize_adapter()
    
    def _initialize_adapter(self):
        """Initialize ChromaDB adapter"""
        try:
            # Set default environment variables if not set
            if not os.getenv("STORAGE_DIR"):
                os.environ["STORAGE_DIR"] = "./storage"
            if not os.getenv("VECTOR_BACKEND"):
                os.environ["VECTOR_BACKEND"] = "chroma"
            
            config = DBConfig.from_env()
            self.adapter = ChromaAdapter(config, collection_name=self.collection_name)
            logger.info(f"Initialized VectorStore with collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing VectorStore: {e}")
            raise
    
    def add_documents(
        self, 
        documents: List[str], 
        embeddings: List[List[float]], 
        metadatas: List[Dict[str, Any]], 
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Add documents to the vector store
        
        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: Optional list of document IDs (auto-generated if not provided)
        
        Returns:
            bool: Success status
        """
        try:
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(documents))]
            
            self.adapter.add_documents(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def search(
        self, 
        query_embedding: List[float], 
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            where: Optional metadata filter
        
        Returns:
            List of search results
        """
        try:
            results = self.adapter.query(
                query_embedding=query_embedding,
                n_results=n_results
            )
            
            # Format results for easier use
            formatted_results = []
            if results and 'documents' in results:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        'content': doc,
                        'score': results.get('distances', [[]])[0][i] if 'distances' in results else 0.0,
                        'metadata': results.get('metadatas', [[]])[0][i] if 'metadatas' in results else {},
                        'id': results.get('ids', [[]])[0][i] if 'ids' in results else f"result_{i}"
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific document by ID
        
        Args:
            doc_id: Document ID
        
        Returns:
            Document data or None if not found
        """
        try:
            # ChromaDB doesn't have a direct get by ID method
            # We'll use search with a very specific filter
            results = self.search([0.0] * 384, n_results=1)  # Dummy embedding
            # This is a limitation - ChromaDB doesn't support direct ID lookup
            # In a real implementation, you might want to maintain a separate ID index
            logger.warning("Direct document retrieval by ID not implemented")
            return None
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return None
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents by IDs
        
        Args:
            ids: List of document IDs to delete
        
        Returns:
            bool: Success status
        """
        try:
            # ChromaDB delete by IDs
            self.adapter.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection
        
        Returns:
            bool: Success status
        """
        try:
            self.adapter.clear_collection()
            logger.info("Cleared collection")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection
        
        Returns:
            Dictionary with collection information
        """
        try:
            # Get collection count and basic info
            count = self.adapter.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "status": "active"
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "status": "error",
                "error": str(e)
            }

# Example usage
if __name__ == "__main__":
    # Test the vector store
    vs = VectorStore("test_collection")
    
    # Test adding documents
    documents = [
        "This is a test document about artificial intelligence.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks to solve complex problems."
    ]
    
    # Mock embeddings (in real usage, use actual embedding model)
    embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
    metadatas = [
        {"source": "test1.txt", "topic": "AI"},
        {"source": "test2.txt", "topic": "ML"},
        {"source": "test3.txt", "topic": "DL"}
    ]
    
    # Add documents
    success = vs.add_documents(documents, embeddings, metadatas)
    print(f"Added documents: {success}")
    
    # Search
    query_embedding = [0.15] * 384  # Mock query embedding
    results = vs.search(query_embedding, n_results=2)
    print(f"Search results: {len(results)}")
    for result in results:
        print(f"- {result['content'][:50]}... (score: {result['score']})")
    
    # Get collection info
    info = vs.get_collection_info()
    print(f"Collection info: {info}")
