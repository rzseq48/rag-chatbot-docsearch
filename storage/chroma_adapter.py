# storage/chroma_adapter.py
from typing import List, Dict, Any
from storage.db_config import DBConfig, get_chroma_client
from chromadb.utils import embedding_functions  # optional depending which embedding func you use

class ChromaAdapter:
    def __init__(self, cfg: DBConfig, collection_name: str = "documents"):
        self.cfg = cfg
        self.client = get_chroma_client(cfg)
        # create or get collection; persist settings depend on chroma installation
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)

    def add_documents(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]], documents: Optional[List[str]] = None):
        """
        Wraps chroma collection.add(...)
        ids: list of document ids (must be unique)
        embeddings: list of vectors
        metadatas: list of dicts (json-serializable)
        documents: optional raw text
        """
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def query(self, query_embedding: List[float], n_results: int = 10):
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        return results
