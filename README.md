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
