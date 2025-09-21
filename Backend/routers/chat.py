from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import uuid

from models.schemas import ChatRequest, ChatResponse
from services.rag_service import process_query
from config import settings

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> Dict[str, Any]:
    """
    Process a chat message and generate a response using RAG.
    
    Args:
        request: Chat request with user message
        
    Returns:
        Response with generated text and sources
    """
    try:
        # Process the query
        result = await process_query(request.message)
        
        # Generate or use existing conversation ID
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Create response
        return {
            "response": result["response"],
            "sources": result["sources"],
            "conversation_id": conversation_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")