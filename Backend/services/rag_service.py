import google.generativeai as genai
from typing import List, Dict, Any
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

from services.embedding_service import embed_query
from services.pinecone_service import similarity_search
from config import settings

# Initialize Google Gemini API
genai.configure(api_key=settings.GOOGLE_API_KEY)

# Text generation model
GENERATION_MODEL = "gemini-2.5-flash"


def format_context(context_chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved context chunks into a single context string.
    
    Args:
        context_chunks: List of context chunks with metadata
        
    Returns:
        Formatted context string
    """
    formatted_context = ""
    
    # Sort chunks by score (highest first)
    sorted_chunks = sorted(context_chunks, key=lambda x: x.get("score", 0), reverse=True)
    
    for i, chunk in enumerate(sorted_chunks):
        formatted_context += f"\n\nCONTEXT CHUNK {i+1} [Source: {chunk.get('source', 'Unknown')}, Page: {chunk.get('page_number', 0)}]:\n"
        formatted_context += chunk.get("text", "")
    
    return formatted_context


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def generate_response(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Generate response using Gemini text model with retrieved context.
    
    Args:
        query: User query
        context_chunks: Retrieved context chunks
        
    Returns:
        Generated response
    """
    # Format context
    context = format_context(context_chunks)
    
    # Create prompt with context and query
    prompt = f"""You are an AI assistant that answers questions based on the provided context.
    
CONTEXT:
{context}

USER QUERY: {query}

Please answer the query based only on the provided context. If the context doesn't contain relevant information to answer the query, 
state that you don't have enough information to provide a complete answer. Do not make up information.
Cite the sources (document name and page number) when providing information from the context.

ANSWER:"""

    # Use asyncio to run the synchronous API call in a thread pool
    try:
        model = genai.GenerativeModel(GENERATION_MODEL)
        response = await asyncio.to_thread(
            model.generate_content,
            prompt
        )
        return response.text
    except Exception as e:
        raise ValueError(f"Error generating response: {str(e)}")


async def process_query(query: str, namespace: str = "default") -> Dict[str, Any]:
    """
    Process user query through the RAG pipeline.
    
    Args:
        query: User query
        namespace: Pinecone namespace
        
    Returns:
        Dictionary with response and source information
    """
    # Create embedding for query
    query_embedding = await embed_query(query)
    
    # Retrieve similar chunks from Pinecone
    context_chunks = await similarity_search(
        query_embedding=query_embedding,
        top_k=settings.TOP_K_RESULTS,
        namespace=namespace
    )
    
    # Generate response using retrieved context
    response_text = await generate_response(query, context_chunks)
    
    return {
        "response": response_text,
        "sources": context_chunks
    }