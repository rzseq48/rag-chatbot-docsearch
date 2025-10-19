# storage/chroma_adapter.py (improved)
from typing import List, Dict, Any, Optional
from storage.db_config import DBConfig, get_chroma_client

class ChromaAdapter:
    def __init__(self, cfg: DBConfig, collection_name: str = "documents"):
        self.cfg = cfg
        self.client = get_chroma_client(cfg)
        self.collection_name = collection_name

        # robustly obtain a collection across chromadb API versions
        self.collection = None
        # try common get_collection signature
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            # try get_or_create_collection if available
            try:
                self.collection = self.client.get_or_create_collection(name=collection_name)
            except Exception:
                # final fallback: create_collection
                try:
                    self.collection = self.client.create_collection(name=collection_name)
                except Exception as exc:
                    raise RuntimeError(f"Failed to get or create chroma collection {collection_name}") from exc

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: Optional[List[str]] = None,
        persist: bool = True
    ):
        """
        Wraps chroma collection.add(...)
        """
        # chroma expects lists; validate lengths
        if not (len(ids) == len(embeddings) == len(metadatas)):
            raise ValueError("ids, embeddings and metadatas must be same length")

        # add documents
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

        # try to persist if supported
        if persist:
            try:
                # some chroma clients have client.persist()
                self.client.persist()
            except Exception:
                # ignore if not supported
                pass

    def query(self, query_embedding: List[float], n_results: int = 10):
        # wrap query consistently (return raw result)
        try:
            results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        except TypeError:
            # other signature: query(embedding_function=..., query_texts=..., etc.)
            results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        return results

    def clear_collection(self):
        """
        Remove all documents from the collection. Implementation depends on chroma version.
        """
        # try delete method
        try:
            self.collection.delete()
        except TypeError:
            # Some versions accept no args; others require ids param
            try:
                self.collection.delete(where={})
            except Exception:
                # last resort: delete & recreate collection
                try:
                    self.client.delete_collection(name=self.collection_name)
                    self.collection = self.client.create_collection(name=self.collection_name)
                except Exception:
                    raise

    def delete_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            # ignore if not supported, or raise if you prefer strictness
            pass
