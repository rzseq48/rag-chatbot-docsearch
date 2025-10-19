"""
Embedding generation for RAG chatbot
"""
import os
import numpy as np
from typing import List, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class Embedder(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass

class MockEmbedder(Embedder):
    """
    Mock embedder for testing - generates random embeddings
    In production, replace with actual embedding model (OpenAI, Sentence-BERT, etc.)
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        logger.info(f"Initialized MockEmbedder with dimension {dimension}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate a random embedding for testing"""
        # In production, this would call an actual embedding model
        np.random.seed(hash(text) % 2**32)  # Deterministic based on text
        return np.random.rand(self.dimension).tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return [self.embed_text(text) for text in texts]

class OpenAIEmbedder(Embedder):
    """
    OpenAI embedding model integration
    Requires OPENAI_API_KEY environment variable
    """
    
    def __init__(self, model: str = "text-embedding-ada-002"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Import here to avoid dependency issues if not installed
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package is required for OpenAIEmbedder")
        
        logger.info(f"Initialized OpenAIEmbedder with model {model}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise

class SentenceBERTEmbedder(Embedder):
    """
    Sentence-BERT embedding model
    Requires sentence-transformers package
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError("sentence-transformers package is required for SentenceBERTEmbedder")
        
        logger.info(f"Initialized SentenceBERTEmbedder with model {model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding using Sentence-BERT"""
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            embeddings = self.model.encode(texts)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise

def get_embedder(embedder_type: str = "mock", **kwargs) -> Embedder:
    """
    Factory function to get appropriate embedder
    
    Args:
        embedder_type: Type of embedder ("mock", "openai", "sentence_bert")
        **kwargs: Additional arguments for embedder initialization
    
    Returns:
        Embedder instance
    """
    if embedder_type == "mock":
        return MockEmbedder(**kwargs)
    elif embedder_type == "openai":
        return OpenAIEmbedder(**kwargs)
    elif embedder_type == "sentence_bert":
        return SentenceBERTEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")

# Example usage
if __name__ == "__main__":
    # Test the embedders
    embedder = get_embedder("mock")
    
    # Test single text
    text = "This is a test document for embedding."
    embedding = embedder.embed_text(text)
    print(f"Single embedding dimension: {len(embedding)}")
    
    # Test batch
    texts = ["Document 1", "Document 2", "Document 3"]
    embeddings = embedder.embed_batch(texts)
    print(f"Batch embeddings: {len(embeddings)} x {len(embeddings[0])}")
