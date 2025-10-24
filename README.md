# 🧠 RAG Chatbot Document Search

A complete AI-powered document retrieval and question-answering system built with FastAPI, ChromaDB, and modern RAG techniques. This production-ready system supports multiple document formats, various LLM providers, and provides both REST API and Python SDK interfaces.

## ✨ Key Features

- **🔍 Multi-Format Document Support**: PDF, DOCX, PPTX, TXT, images, and archives with intelligent text extraction
- **🤖 Multiple LLM Providers**: OpenAI GPT, Anthropic Claude, Groq (ultra-fast inference), or mock providers for testing
- **🧮 Advanced Embeddings**: OpenAI, Sentence-BERT, or mock embedding providers
- **📊 Vector Search**: ChromaDB-powered semantic search with similarity scoring
- **🔧 OCR Support**: Tesseract integration for scanned documents and images
- **🌐 RESTful API**: Complete FastAPI application with automatic Swagger documentation
- **🐳 Docker Ready**: Production-ready containerization with Docker Compose
- **🧪 Comprehensive Testing**: Full test suite with mock providers for development
- **📚 Complete Documentation**: Extensive examples and API documentation

## 🏗️ Architecture

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
│   (PDF/DOCX/    │    │   (Overlap &    │    │   (ChromaDB)    │
│    Images/OCR)  │    │    Metadata)    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Query         │◀───│   RAG           │◀───│   Similarity    │
│   Processing    │    │   Pipeline      │    │   Search        │
│   (Natural      │    │   (Context +    │    │   (Vector       │
│    Language)    │    │    Generation)  │    │    Matching)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag-chatbot-docsearch

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp env.example .env
```

### 2. Configuration

Edit the `.env` file to configure your providers:

```bash
# For OpenAI (requires API key)
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDER_TYPE=openai
LLM_PROVIDER=openai

# For Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_api_key_here
LLM_PROVIDER=anthropic

# For Groq (ultra-fast inference)
GROQ_API_KEY=your_groq_api_key_here
LLM_PROVIDER=groq

# Or use mock providers for testing (no API keys needed)
EMBEDDER_TYPE=mock
LLM_PROVIDER=mock
```

### 3. Run the Demo

```bash
# Run the complete demo
python demo.py

# Or start the API server
python api/main.py
```

### 4. Docker Deployment

```bash
# Start with Docker Compose
docker-compose up -d

# Or build and run manually
docker build -t rag-chatbot .
docker run -p 8000:8000 rag-chatbot
```

## 📖 Usage

### Python SDK

```python
from rag_pipeline import RAGPipeline

# Initialize RAG system
rag = RAGPipeline(
    storage_dir="./storage",
    embedder_type="openai",  # or "mock", "sentence_bert"
    llm_provider="groq"      # or "mock", "openai", "anthropic"
)

# Ingest documents
result = rag.ingest_documents("./documents/")
print(f"Processed {result['total_documents']} documents")

# Query the system
response = rag.query("What is machine learning?")
print(f"Answer: {response['answer']}")
print(f"Sources: {len(response['sources'])}")
print(f"Confidence: {response['confidence']:.2f}")
```

### REST API

#### Upload Documents
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

#### Query Documents
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?", 
    "max_results": 5
  }'
```

#### Ingest from Directory
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "/path/to/documents",
    "chunk_size": 1000
  }'
```

#### Health Check
```bash
curl http://localhost:8000/health
```

### API Endpoints

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/` | GET | Health check | - |
| `/health` | GET | Service status | - |
| `/upload` | POST | Upload document | `file` (multipart) |
| `/query` | POST | Query documents | `query`, `max_results` |
| `/ingest` | POST | Ingest from path | `input_path`, `chunk_size` |
| `/documents` | GET | List documents | - |
| `/documents` | DELETE | Clear collection | - |

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STORAGE_DIR` | `./storage` | ChromaDB storage directory |
| `VECTOR_BACKEND` | `chroma` | Vector database backend |
| `EMBEDDER_TYPE` | `mock` | Embedding provider (`mock`, `openai`, `sentence_bert`) |
| `LLM_PROVIDER` | `mock` | LLM provider (`mock`, `openai`, `anthropic`, `groq`) |
| `GROQ_API_KEY` | - | Groq API key for ultra-fast inference |
| `CHUNK_SIZE` | `1000` | Document chunk size |
| `CHUNK_OVERLAP` | `200` | Chunk overlap |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |

### Supported File Types

- **Text Files**: `.txt`, `.md`, `.csv`, `.log`
- **PDF Documents**: `.pdf` (with OCR fallback for scanned documents)
- **Office Documents**: `.docx`, `.pptx`, `.xlsx`
- **Images**: `.jpg`, `.png`, `.tiff`, `.bmp` (with OCR text extraction)
- **Archives**: `.zip`, `.tar`, `.tar.gz` (with recursive processing)

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_rag_pipeline.py

# Run with coverage
python -m pytest --cov=. tests/

# Run the demo
python demo.py

# Test API endpoints
python test_api.py
```

## 📁 Project Structure

```
rag-chatbot-docsearch/
├── api/                    # FastAPI application
│   └── main.py            # API endpoints and server
├── data_ingestion/        # Document processing pipeline
│   ├── loader.py          # Main loader orchestrator
│   ├── cleaner.py         # Text cleaning utilities
│   ├── ocr_reader.py      # OCR processing
│   └── pipeline.py        # Processing pipeline
├── embeddings/            # Embedding generation
│   └── embedder.py        # Multiple embedding providers
├── llm/                   # LLM integration
│   └── llm_provider.py    # Multiple LLM providers
├── storage/               # Database layer
│   ├── db_config.py       # Database configuration
│   ├── chroma_adapter.py  # ChromaDB adapter
│   ├── raw_store.py       # Raw document storage
│   └── gold_store.py      # Processed document storage
├── vectorstore/           # Vector operations
│   └── chroma_store.py    # Vector store wrapper
├── utils/                 # Utilities
│   └── logger.py          # Logging configuration
├── tests/                 # Test suite
│   ├── test_rag_pipeline.py
│   ├── test_chroma_adapter_smoke.py
│   └── test_loader_*.py
├── examples/              # Usage examples
│   ├── chromadb_usage.py
│   ├── complete_example.py
│   └── groq_example.py
├── notebooks/             # Jupyter notebooks
│   └── test.ipynb
├── rag_pipeline.py       # Complete RAG pipeline
├── demo.py               # Interactive demo
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose setup
├── env.example          # Environment template
├── CHROMADB_SETUP.md    # ChromaDB setup guide
├── PROJECT_COMPLETION.md # Project completion status
└── README.md            # This file
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

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

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

# Run container with volume mounts
docker run -p 8000:8000 \
  -v $(pwd)/storage:/app/storage \
  -v $(pwd)/data:/app/data \
  rag-chatbot
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
   # Windows:
   # Download from: https://github.com/UB-Mannheim/tesseract/wiki
   ```

4. **Memory Issues**
   ```bash
   # Reduce chunk size in .env
   CHUNK_SIZE=500
   ```

5. **API Key Issues**
   ```bash
   # Check your .env file has correct API keys
   # Use mock providers for testing without API keys
   EMBEDDER_TYPE=mock
   LLM_PROVIDER=mock
   ```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🎯 Use Cases

- **Document Q&A**: Ask questions about uploaded documents
- **Knowledge Base**: Build searchable knowledge bases from documents
- **Research Assistant**: Help researchers find relevant information
- **Customer Support**: Answer questions based on documentation
- **Content Analysis**: Analyze and summarize document content

## 🔗 Related Documentation

- [ChromaDB Setup Guide](CHROMADB_SETUP.md) - Detailed ChromaDB configuration
- [Project Completion Status](PROJECT_COMPLETION.md) - Implementation details
- [Issues and Fixes](ISSUES_AND_FIXES.md) - Known issues and solutions

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



## 🎉 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) for the API
- Powered by [ChromaDB](https://www.trychroma.com/) for vector storage
- Uses [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for document processing
- Supports multiple LLM providers for flexibility