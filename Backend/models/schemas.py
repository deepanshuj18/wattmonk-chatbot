from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Model for chat request from user."""
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for tracking context")


class RetrievedChunk(BaseModel):
    """Model for retrieved context chunk with metadata."""
    text: str = Field(..., description="Text content of the chunk")
    source: str = Field(..., description="Source document name")
    page_number: int = Field(..., description="Page number in the source document")
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    score: float = Field(..., description="Similarity score")


class ChatResponse(BaseModel):
    """Model for chat response to user."""
    response: str = Field(..., description="Generated response text")
    sources: List[RetrievedChunk] = Field([], description="Source chunks used for generation")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for tracking context")


class PDFUploadResponse(BaseModel):
    """Model for PDF upload response."""
    filename: str = Field(..., description="Name of the uploaded file")
    pages_processed: int = Field(..., description="Number of pages processed")
    chunks_created: int = Field(..., description="Number of text chunks created")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")


class IndexStats(BaseModel):
    """Model for Pinecone index statistics."""
    index_name: str = Field(..., description="Name of the Pinecone index")
    vector_count: int = Field(..., description="Number of vectors in the index")
    dimension: int = Field(..., description="Dimension of vectors")
    namespaces: Dict[str, int] = Field({}, description="Namespaces and their vector counts")


class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str = Field(..., description="Service status")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Status of individual components")