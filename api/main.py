"""
FastAPI application for RAG Chatbot Document Search
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import logging
import tempfile
from pathlib import Path

# Import RAG pipeline
import sys
sys.path.append(str(Path(__file__).parent.parent))
from rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot Document Search",
    description="AI-powered document retrieval and question-answering system",
    version="1.0.0"
)

# Initialize RAG pipeline
rag_pipeline = None

def get_rag_pipeline():
    """Get or create RAG pipeline instance"""
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline(
            storage_dir="./storage",
            embedder_type=os.getenv("EMBEDDER_TYPE", "mock"),
            llm_provider=os.getenv("LLM_PROVIDER", "mock"),
            collection_name="documents"
        )
    return rag_pipeline

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 10

class QueryResponse(BaseModel):
    results: List[dict]
    query: str
    total_results: int
    answer: Optional[str] = ""
    confidence: Optional[float] = 0.0

class HealthResponse(BaseModel):
    status: str
    message: str

# Routes
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="healthy",
        message="RAG Chatbot Document Search API is running"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Service is operational"
    )

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for processing"""
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Process document with RAG pipeline
            rag = get_rag_pipeline()
            result = rag.ingest_documents(tmp_file_path)
            
            return {
                "message": f"File {file.filename} processed successfully",
                "filename": file.filename,
                "content_type": file.content_type,
                "processing_result": result
            }
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using natural language"""
    try:
        logger.info(f"Received query: {request.query}")
        
        # Process query with RAG pipeline
        rag = get_rag_pipeline()
        response = rag.query(
            question=request.query,
            max_results=request.max_results
        )
        
        # Format sources for response
        sources = []
        for source in response.get('sources', []):
            sources.append({
                "content": source['content'],
                "source": source.get('metadata', {}).get('file_path', 'unknown'),
                "score": source['score']
            })
        
        return QueryResponse(
            results=sources,
            query=request.query,
            total_results=len(sources),
            answer=response.get('answer', ''),
            confidence=response.get('confidence', 0.0)
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all processed documents"""
    try:
        rag = get_rag_pipeline()
        info = rag.get_collection_info()
        
        return {
            "documents": [],  # Could be enhanced to list actual documents
            "total": info.get("document_count", 0),
            "collection_name": info.get("collection_name", "documents"),
            "status": info.get("status", "unknown")
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_documents(
    input_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """Ingest documents from a directory or file path"""
    try:
        rag = get_rag_pipeline()
        result = rag.ingest_documents(input_path, chunk_size, chunk_overlap)
        
        return {
            "message": "Document ingestion completed",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error ingesting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the collection"""
    try:
        rag = get_rag_pipeline()
        success = rag.clear_collection()
        
        return {
            "message": "Documents cleared successfully" if success else "Failed to clear documents",
            "success": success
        }
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
