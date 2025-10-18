# ChromaDB Setup Guide for RAG Chatbot

This guide explains how to set up and use ChromaDB in your RAG chatbot project.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install ChromaDB specifically
pip install chromadb>=0.5,<1.0
```

### 2. Environment Configuration

Create a `.env` file in your project root:

```bash
# Copy the example
cp env.example .env
```

Key environment variables:
- `STORAGE_DIR`: Directory for ChromaDB storage (default: `./storage`)
- `VECTOR_BACKEND`: Set to `"chroma"` to use ChromaDB
- `SQLITE_FILENAME`: SQLite database for metadata

### 3. Basic Usage

```python
import os
from storage.db_config import DBConfig
from storage.chroma_adapter import ChromaAdapter

# Set environment variables
os.environ["STORAGE_DIR"] = "./storage"
os.environ["VECTOR_BACKEND"] = "chroma"

# Create configuration
config = DBConfig.from_env()

# Initialize ChromaDB adapter
adapter = ChromaAdapter(config, collection_name="documents")

# Add documents
adapter.add_documents(
    ids=["doc1", "doc2"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    metadatas=[{"source": "file1"}, {"source": "file2"}],
    documents=["Document 1 content", "Document 2 content"]
)

# Query documents
results = adapter.query(query_embedding=[0.1, 0.2, ...], n_results=5)
```

## 📁 Project Structure

Your ChromaDB integration is organized as follows:

```
storage/
├── db_config.py          # Database configuration and ChromaDB client setup
├── chroma_adapter.py     # ChromaDB adapter with CRUD operations
└── chroma/               # ChromaDB persistent storage (created automatically)

vectorstore/
└── chroma_store.py       # Vector store implementation

tests/
└── test_chroma_adapter_smoke.py  # ChromaDB smoke tests
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STORAGE_DIR` | `./storage` | Base directory for all storage |
| `VECTOR_BACKEND` | `chroma` | Vector database backend |
| `SQLITE_FILENAME` | `storage_meta.sqlite` | SQLite metadata database |
| `VECTOR_SETTINGS` | `{}` | JSON string with additional settings |

### ChromaDB Settings

You can pass additional settings via `VECTOR_SETTINGS`:

```bash
VECTOR_SETTINGS='{"collection_name": "my_docs", "distance_metric": "cosine"}'
```

## 🧪 Testing

### Run the Smoke Test

```bash
python tests/test_chroma_adapter_smoke.py
```

### Run the Usage Example

```bash
python examples/chromadb_usage.py
```

## 📚 API Reference

### ChromaAdapter Class

#### `__init__(cfg: DBConfig, collection_name: str = "documents")`
Initialize ChromaDB adapter with configuration and collection name.

#### `add_documents(ids, embeddings, metadatas, documents=None, persist=True)`
Add documents to the collection.

**Parameters:**
- `ids`: List of document IDs
- `embeddings`: List of embedding vectors
- `metadatas`: List of metadata dictionaries
- `documents`: Optional list of document texts
- `persist`: Whether to persist changes

#### `query(query_embedding: List[float], n_results: int = 10)`
Query for similar documents.

**Parameters:**
- `query_embedding`: Query vector
- `n_results`: Number of results to return

#### `clear_collection()`
Remove all documents from the collection.

#### `delete_collection()`
Delete the entire collection.

## 🔍 Integration with RAG Pipeline

Your ChromaDB setup integrates with the broader RAG pipeline:

1. **Data Ingestion** (`data_ingestion/`): Documents are processed and prepared
2. **Embeddings** (`embeddings/`): Text is converted to vectors
3. **Storage** (`storage/`): ChromaDB stores vectors and metadata
4. **Retrieval**: Similar documents are retrieved for RAG

## 🛠️ Troubleshooting

### Common Issues

1. **ImportError: chromadb not installed**
   ```bash
   pip install chromadb
   ```

2. **Permission errors**
   - Ensure write permissions to `STORAGE_DIR`
   - Check if directory exists and is accessible

3. **Collection not found**
   - ChromaAdapter automatically creates collections
   - Check collection name spelling

4. **Embedding dimension mismatch**
   - Ensure all embeddings have the same dimension
   - Use consistent embedding models

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🚀 Advanced Usage

### Custom Collection Settings

```python
# Create collection with custom settings
config = DBConfig.from_env()
adapter = ChromaAdapter(config, collection_name="custom_docs")

# Add with custom metadata
adapter.add_documents(
    ids=["doc1"],
    embeddings=[[0.1, 0.2, 0.3]],
    metadatas=[{"source": "pdf", "page": 1, "section": "introduction"}],
    documents=["Document content"]
)
```

### Batch Operations

```python
# Add multiple documents efficiently
batch_size = 100
for i in range(0, len(all_documents), batch_size):
    batch = all_documents[i:i+batch_size]
    adapter.add_documents(
        ids=batch["ids"],
        embeddings=batch["embeddings"],
        metadatas=batch["metadatas"],
        documents=batch["documents"]
    )
```

## 📖 Next Steps

1. **Set up embeddings**: Configure your embedding model in `embeddings/embedder.py`
2. **Integrate with pipeline**: Use ChromaDB in your data ingestion pipeline
3. **Add retrieval logic**: Implement similarity search in your RAG system
4. **Monitor performance**: Add logging and metrics for production use

## 🔗 Related Files

- `storage/db_config.py`: Database configuration
- `storage/chroma_adapter.py`: ChromaDB operations
- `tests/test_chroma_adapter_smoke.py`: Basic tests
- `examples/chromadb_usage.py`: Usage examples
