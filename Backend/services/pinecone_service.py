import asyncio
import pinecone
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from config import settings

# Initialize Pinecone client
pinecone.init(
    api_key=settings.PINECONE_API_KEY,
    environment=settings.PINECONE_ENVIRONMENT
)

# Global variable to store the index
_index = None


async def get_index():
    """
    Get or create Pinecone index.
    
    Returns:
        Pinecone index instance
    """
    global _index
    
    if _index is not None:
        return _index
    
    # Check if index exists
    if settings.PINECONE_INDEX_NAME not in pinecone.list_indexes():
        # Create index if it doesn't exist
        pinecone.create_index(
            name=settings.PINECONE_INDEX_NAME,
            dimension=768,  # Dimension for Gemini embeddings
            metric="cosine"
        )
    
    # Connect to index
    _index = pinecone.Index(settings.PINECONE_INDEX_NAME)
    return _index


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def store_embeddings(chunks_with_embeddings: List[Dict[str, Any]], namespace: str = "default") -> Dict[str, Any]:
    """
    Store embeddings in Pinecone.
    
    Args:
        chunks_with_embeddings: List of chunks with embeddings
        namespace: Pinecone namespace
        
    Returns:
        Upsert response
    """
    index = await get_index()
    
    # Prepare vectors for upsert
    vectors = []
    for chunk in chunks_with_embeddings:
        # Extract metadata (excluding the embedding and text to avoid size limits)
        metadata = {
            "text": chunk["text"],
            "source": chunk["source"],
            "page_number": chunk["page_number"],
            "chunk_id": chunk["chunk_id"]
        }
        
        vectors.append({
            "id": chunk["chunk_id"],
            "values": chunk["embedding"],
            "metadata": metadata
        })
    
    # Upsert in batches to avoid size limits
    batch_size = 100
    results = {"upserted_count": 0}
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        
        try:
            # Use asyncio to run the synchronous API call in a thread pool
            response = await asyncio.to_thread(
                index.upsert,
                vectors=batch,
                namespace=namespace
            )
            
            results["upserted_count"] += response.get("upserted_count", 0)
        except Exception as e:
            if "not found" in str(e) or "Forbidden" in str(e):
                raise ValueError("Invalid Pinecone API key or environment. Please check your .env file.")
            raise ValueError(f"Error storing embeddings in Pinecone: {str(e)}")
    
    return results


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def similarity_search(query_embedding: List[float], top_k: int = None, namespace: str = "default") -> List[Dict[str, Any]]:
    """
    Search for similar vectors in Pinecone.
    
    Args:
        query_embedding: Query embedding vector
        top_k: Number of results to return
        namespace: Pinecone namespace
        
    Returns:
        List of similar chunks with metadata and scores
    """
    index = await get_index()
    
    # Use settings if not explicitly provided
    top_k = top_k or settings.TOP_K_RESULTS
    
    # Use asyncio to run the synchronous API call in a thread pool
    response = await asyncio.to_thread(
        index.query,
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    
    results = []
    for match in response.get("matches", []):
        # Extract metadata and score
        metadata = match.get("metadata", {})
        results.append({
            "text": metadata.get("text", ""),
            "source": metadata.get("source", ""),
            "page_number": metadata.get("page_number", 0),
            "chunk_id": metadata.get("chunk_id", ""),
            "score": match.get("score", 0.0)
        })
    
    return results


async def get_index_stats(namespace: Optional[str] = None) -> Dict[str, Any]:
    """
    Get statistics about the Pinecone index.
    
    Args:
        namespace: Optional namespace to filter stats
        
    Returns:
        Dictionary with index statistics
    """
    index = await get_index()
    
    # Use asyncio to run the synchronous API call in a thread pool
    stats = await asyncio.to_thread(index.describe_index_stats)
    
    result = {
        "index_name": settings.PINECONE_INDEX_NAME,
        "vector_count": stats.get("total_vector_count", 0),
        "dimension": stats.get("dimension", 768),
        "namespaces": stats.get("namespaces", {})
    }
    
    return result
