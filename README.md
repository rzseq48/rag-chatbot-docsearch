# 🧠 rag-chatbot-docsearch

An AI-powered document retrieval and question-answering system built with FastAPI, LlamaIndex, and ChromaDB.

---

## 🚀 Overview
rag-chatbot-docsearch ingests documents (e.g., PDFs, text files), converts them into embeddings, and allows natural-language querying using a retrieval-augmented generation (RAG) pipeline.

---

## 🧩 Architecture

**Modules**
- `data_ingestion/` → load & chunk documents  
- `embeddings/` → generate and store embeddings  
- `retrieval/` → search vector DB  
- `api/` → expose endpoints for query and ingestion  

---

## ⚙️ Tech Stack
- **Backend:** FastAPI  
- **LLM/RAG:** LlamaIndex, OpenAI API  
- **Vector Store:** ChromaDB  
- **Environment:** Python 3.10+  
- **Deployment:** Docker + Render (planned)

---

## 🧭 Setup Instructions
```bash
git clone git@github.com:rzseq48/rag-chatbot-docsearch.git
cd rag-chatbot-docsearch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Set up your environment variables in .env:
OPENAI_API_KEY=your_key

Run the FastAPI app:
uvicorn app.main:app --reload
