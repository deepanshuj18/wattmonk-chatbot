from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from models.schemas import HealthResponse, IndexStats
from services.pinecone_service import get_index_stats
from config import settings

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> Dict[str, Any]:
    """
    Check the health of the service and its components.
    
    Returns:
        Health status of the service
    """
    try:
        # Check Pinecone connection
        pinecone_status = "ok"
        pinecone_details = {"status": "connected"}
        try:
            await get_index_stats()
        except Exception as e:
            pinecone_status = "error"
            pinecone_details = {"status": "error", "message": str(e)}
        
        # Check Google API connection
        google_status = "ok"
        google_details = {"status": "connected"}
        if not settings.GOOGLE_API_KEY:
            google_status = "error"
            google_details = {"status": "error", "message": "API key not configured"}
        
        return {
            "status": "healthy" if pinecone_status == "ok" and google_status == "ok" else "degraded",
            "components": {
                "pinecone": pinecone_details,
                "google_ai": google_details,
                "api": {"status": "ok"}
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking health: {str(e)}")


@router.get("/index-stats", response_model=IndexStats)
async def index_stats(namespace: str = None) -> Dict[str, Any]:
    """
    Get statistics about the Pinecone index.
    
    Args:
        namespace: Optional namespace to filter stats
        
    Returns:
        Index statistics
    """
    try:
        return await get_index_stats(namespace)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting index stats: {str(e)}")