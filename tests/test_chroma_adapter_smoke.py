# tests/test_chroma_adapter_smoke.py
import os
import shutil
import tempfile
import random
import numpy as np

# ensure your package path is importable if running outside package root
import sys
sys.path.append(str(os.path.abspath(".")))

from storage.db_config import DBConfig
from storage.chroma_adapter import ChromaAdapter

def make_random_embedding(dim=1536):
    # use small dim for test to be lightweight
    return [float(x) for x in np.random.rand(dim).tolist()]

def run_smoke_test():
    tmpdir = tempfile.mkdtemp(prefix="chroma_test_")
    print("Using temporary STORAGE_DIR:", tmpdir)

    # set env vars for DBConfig.from_env
    os.environ["STORAGE_DIR"] = tmpdir
    os.environ["VECTOR_BACKEND"] = "chroma"

    cfg = DBConfig.from_env()
    print("DBG: cfg.base_dir:", cfg.base_dir)

    # instantiate adapter
    adapter = ChromaAdapter(cfg, collection_name="test_documents")

    # add 3 small test documents with small-dim embeddings to keep test fast
    dim = 16  # small dim makes the test quick
    ids = ["t1", "t2", "t3"]
    docs = ["AI is new electricity.", "Chroma stores embeddings.", "This is a third test doc."]
    embeddings = [[random.random() for _ in range(dim)] for _ in ids]
    metadatas = [{"source": "smoke"} for _ in ids]

    print("Adding documents ...")
    adapter.add_documents(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=docs)
    print("Added documents.")

    # query using the embedding of doc 1 and print results
    query_emb = embeddings[0]
    print("Querying ...")
    results = adapter.query(query_embedding=query_emb, n_results=3)
    print("Query results (raw):", results)

    # cleanup
    try:
        adapter.delete_collection()
        print("Collection deleted.")
    except Exception as e:
        print("Warning: could not delete collection:", e)

    # remove tmpdir
    shutil.rmtree(tmpdir)
    print("Cleaned up test dir.")

if __name__ == "__main__":
    run_smoke_test()
