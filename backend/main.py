import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from conversational_agent import ConversationalAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Conversational Agent API",
    description="A conversational agent that can answer questions about uploaded documents using RAG with OpenAI integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get OpenAI API key from environment
openai_api_key = os.getenv('OPENAI_API_KEY')
use_openai = bool(openai_api_key)

if use_openai:
    logger.info("OpenAI API key found and enabled for enhanced answer generation")
else:
    logger.info("OpenAI API key not found. Using local RAG system for answer generation")

# Initialize the conversational agent
agent = ConversationalAgent(
    openai_api_key=openai_api_key,
    use_openai=use_openai
)

# Create uploads directory
UPLOADS_DIR = Path("./uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str
    use_history: bool = True

class QuestionResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    error: Optional[str] = None
    confidence: Optional[float] = None
    context_used: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    answer_method: Optional[str] = None

class DocumentUploadResponse(BaseModel):
    success: bool
    filename: Optional[str] = None
    chunks_created: Optional[int] = None
    word_count: Optional[int] = None
    file_size: Optional[int] = None
    error: Optional[str] = None

class StatsResponse(BaseModel):
    documents: Dict[str, Any]
    supported_formats: List[str]
    openai_enabled: bool

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Document Conversational Agent API",
        "version": "1.0.0",
        "status": "running",
        "openai_enabled": use_openai
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "agent_initialized": True,
        "openai_enabled": use_openai
    }

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document to the agent's knowledge base
    
    Supported formats: PDF, DOCX, TXT, MD, JPG, PNG, TIFF
    """
    try:
        # Validate file format
        supported_formats = agent.get_supported_formats()
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: {', '.join(supported_formats)}"
            )
        
        # Save uploaded file
        file_path = UPLOADS_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File uploaded: {file_path}")
        
        # Process document with agent
        result = agent.add_document(str(file_path))
        
        if result['success']:
            return DocumentUploadResponse(
                success=True,
                filename=result['filename'],
                chunks_created=result['chunks_created'],
                word_count=result['word_count'],
                file_size=result['file_size']
            )
        else:
            raise HTTPException(status_code=500, detail=result['error'])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the uploaded documents
    """
    try:
        result = agent.ask_question(request.question, request.use_history)
        
        if result['success']:
            return QuestionResponse(
                success=True,
                answer=result['answer'],
                confidence=result.get('confidence'),
                context_used=result.get('context_used'),
                metadata=result.get('metadata'),
                answer_method=result.get('answer_method')
            )
        else:
            return QuestionResponse(
                success=False,
                error=result.get('error', 'Unknown error occurred'),
                answer_method=result.get('answer_method')
            )
            
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/conversation-history")
async def get_conversation_history():
    """Get the conversation history"""
    try:
        history = agent.get_conversation_history()
        return {"history": history}
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation history: {str(e)}")

@app.delete("/conversation-history")
async def clear_conversation_history():
    """Clear the conversation history"""
    try:
        agent.clear_conversation_history()
        return {"message": "Conversation history cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing conversation history: {str(e)}")

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get statistics about loaded documents and supported formats"""
    try:
        document_stats = agent.get_document_stats()
        supported_formats = agent.get_supported_formats()
        
        return StatsResponse(
            documents=document_stats,
            supported_formats=supported_formats,
            openai_enabled=use_openai
        )
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")

@app.delete("/documents")
async def clear_all_documents():
    """Clear all documents from the knowledge base"""
    try:
        agent.clear_all_documents()
        
        # Clean up uploaded files
        for file_path in UPLOADS_DIR.iterdir():
            if file_path.is_file():
                file_path.unlink()
        
        return {"message": "All documents and conversation history cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")

@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported document formats"""
    try:
        formats = agent.get_supported_formats()
        return {"supported_formats": formats}
    except Exception as e:
        logger.error(f"Error getting supported formats: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving supported formats: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 