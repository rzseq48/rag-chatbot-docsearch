# 🧠 RAG Chatbot Document Search

A complete AI-powered document retrieval and question-answering system built with FastAPI, ChromaDB, and modern RAG techniques.

## 🚀 Features

- **Multi-format Document Support**: PDF, DOCX, PPTX, TXT, images, and archives
- **Intelligent Text Extraction**: OCR support for images and scanned documents
- **Vector Search**: ChromaDB-powered semantic search
- **Multiple LLM Providers**: OpenAI, Anthropic, Groq, or mock providers
- **RESTful API**: Complete FastAPI application with Swagger docs
- **Docker Support**: Ready for production deployment
- **Comprehensive Testing**: Full test suite included

## 🧩 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Data          │    │   Embeddings    │
│   Ingestion     │───▶│   Processing   │───▶│   Generation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File          │    │   Text          │    │   Vector        │
│   Handlers      │    │   Chunking      │    │   Storage       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Query         │◀───│   RAG           │◀───│   Similarity    │
│   Processing    │    │   Pipeline      │    │   Search        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## ⚙️ Tech Stack

- **Backend**: FastAPI + Uvicorn
- **Vector Store**: ChromaDB
- **Embeddings**: OpenAI, Sentence-BERT, or Mock
- **LLM**: OpenAI GPT, Anthropic Claude, Groq (fast inference), or Mock
- **Document Processing**: PyPDF, python-docx, python-pptx, Tesseract OCR
- **Storage**: SQLite + ChromaDB
- **Deployment**: Docker + Docker Compose
- **Environment**: Python 3.10+

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag-chatbot-docsearch

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp env.example .env

# Edit configuration
nano .env
```

### 3. Run the Application

```bash
# Start the API server
python api/main.py

# Or use Docker
docker-compose up
```

### 4. Test the System

```bash
# Run tests
python -m pytest tests/

# Test the API
curl http://localhost:8000/health
```

## 📖 Usage

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Service status |
| `/upload` | POST | Upload document |
| `/query` | POST | Query documents |
| `/ingest` | POST | Ingest from path |
| `/documents` | GET | List documents |
| `/documents` | DELETE | Clear collection |

### Example API Usage

```bash
# Upload a document
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"

# Query documents
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is artificial intelligence?", "max_results": 5}'

# Ingest documents from directory
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"input_path": "/path/to/documents", "chunk_size": 1000}'
```

### Python Usage

```python
from rag_pipeline import RAGPipeline

# Initialize RAG system
rag = RAGPipeline(
    storage_dir="./storage",
    embedder_type="mock",  # or "openai", "sentence_bert"
    llm_provider="mock"    # or "openai", "anthropic", "groq"
)

# Ingest documents
result = rag.ingest_documents("./documents/")
print(f"Processed {result['total_documents']} documents")

# Query the system
response = rag.query("What is machine learning?")
print(f"Answer: {response['answer']}")
print(f"Sources: {len(response['sources'])}")
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STORAGE_DIR` | `./storage` | ChromaDB storage directory |
| `VECTOR_BACKEND` | `chroma` | Vector database backend |
| `EMBEDDER_TYPE` | `mock` | Embedding provider |
| `LLM_PROVIDER` | `mock` | LLM provider |
| `CHUNK_SIZE` | `1000` | Document chunk size |
| `CHUNK_OVERLAP` | `200` | Chunk overlap |

### Supported File Types

- **Text**: `.txt`, `.md`, `.csv`, `.log`
- **PDF**: `.pdf` (with OCR fallback)
- **Office**: `.docx`, `.pptx`, `.xlsx`
- **Images**: `.jpg`, `.png`, `.tiff`, `.bmp` (with OCR)
- **Archives**: `.zip`, `.tar`, `.tar.gz`

## 🐳 Docker Deployment

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Using Docker

```bash
# Build image
docker build -t rag-chatbot .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/storage:/app/storage \
  -v $(pwd)/data:/app/data \
  rag-chatbot
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_rag_pipeline.py

# Run with coverage
python -m pytest --cov=. tests/
```

## 📁 Project Structure

```
rag-chatbot-docsearch/
├── api/                    # FastAPI application
│   └── main.py            # API endpoints
├── data_ingestion/        # Document processing
│   ├── loader.py          # Main loader orchestrator
│   ├── cleaner.py         # Text cleaning utilities
│   └── ocr_reader.py      # OCR processing
├── embeddings/            # Embedding generation
│   └── embedder.py        # Embedding providers
├── llm/                   # LLM integration
│   └── llm_provider.py    # LLM providers
├── storage/               # Database layer
│   ├── db_config.py       # Database configuration
│   ├── chroma_adapter.py  # ChromaDB adapter
│   └── ...
├── vectorstore/           # Vector operations
│   └── chroma_store.py   # Vector store wrapper
├── tests/                 # Test suite
├── examples/             # Usage examples
├── rag_pipeline.py       # Complete RAG pipeline
├── requirements.txt      # Dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose setup
└── README.md           # This file
```

## 🔌 Extending the System

### Adding New Document Handlers

```python
from data_ingestion.loader import Handler

class CustomHandler(Handler):
    def inspect(self, fd):
        return {"page_count": 1, "has_embedded_text": True}
    
    def extract(self, fd):
        # Your extraction logic
        yield {"content": "extracted text", "metadata": {}}
```

### Adding New Embedding Providers

```python
from embeddings.embedder import Embedder

class CustomEmbedder(Embedder):
    def embed_text(self, text):
        # Your embedding logic
        return [0.1] * 384
    
    def embed_batch(self, texts):
        return [self.embed_text(text) for text in texts]
```

### Adding New LLM Providers

```python
from llm.llm_provider import LLMProvider

class CustomLLMProvider(LLMProvider):
    def generate_response(self, prompt, context=None):
        # Your LLM logic
        return "Generated response"
```

## 🚨 Troubleshooting

### Common Issues

1. **ChromaDB Connection Error**
   ```bash
   # Ensure storage directory exists
   mkdir -p storage
   ```

2. **Missing Dependencies**
   ```bash
   # Install all requirements
   pip install -r requirements.txt
   ```

3. **OCR Not Working**
   ```bash
   # Install Tesseract
   # Ubuntu/Debian:
   sudo apt-get install tesseract-ocr
   # macOS:
   brew install tesseract
   ```

4. **Memory Issues**
   ```bash
   # Reduce chunk size in .env
   CHUNK_SIZE=500
   ```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For questions and support, please open an issue on GitHub.
