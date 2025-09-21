import asyncio
import google.generativeai as genai
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from config import settings

# Initialize Google Gemini API
genai.configure(api_key=settings.GOOGLE_API_KEY)

# Embedding model
EMBEDDING_MODEL = "models/embedding-001"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def create_embedding(text: str) -> List[float]:
    """
    Create embedding for a single text using Google Gemini.
    
    Args:
        text: Text to create embedding for
        
    Returns:
        Embedding vector as list of floats
    """
    try:
        # Use asyncio to run the synchronous API call in a thread pool
        embedding = await asyncio.to_thread(
            genai.embed_content,
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        
        # Return the embedding values
        return embedding["embedding"]
    except Exception as e:
        if "API key not valid" in str(e):
            raise ValueError("Invalid Google API key. Please check your .env file.")
        raise ValueError(f"Error creating embedding: {str(e)}")


async def create_embeddings_batch(texts: List[str], batch_size: int = 5) -> List[List[float]]:
    """
    Create embeddings for multiple texts in batches.
    
    Args:
        texts: List of texts to create embeddings for
        batch_size: Number of texts to process in each batch
        
    Returns:
        List of embedding vectors
    """
    all_embeddings = []
    
    # Process in batches to avoid rate limits
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Create tasks for concurrent processing
        tasks = [create_embedding(text) for text in batch]
        batch_embeddings = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for exceptions
        for j, embedding in enumerate(batch_embeddings):
            if isinstance(embedding, Exception):
                print(f"Error creating embedding for text {i+j}: {str(embedding)}")
                # Retry the failed embedding
                try:
                    embedding = await create_embedding(batch[j])
                    all_embeddings.append(embedding)
                except Exception as e:
                    print(f"Retry failed for text {i+j}: {str(e)}")
                    # Add a placeholder embedding (zeros)
                    all_embeddings.append([0.0] * 768)  # Typical embedding dimension
            else:
                all_embeddings.append(embedding)
    
    return all_embeddings


async def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create embeddings for text chunks.
    
    Args:
        chunks: List of chunk dictionaries with text and metadata
        
    Returns:
        List of chunks with embeddings added
    """
    # Extract text from chunks
    texts = [chunk["text"] for chunk in chunks]
    
    # Create embeddings
    embeddings = await create_embeddings_batch(texts)
    
    # Add embeddings to chunks
    for i, embedding in enumerate(embeddings):
        chunks[i]["embedding"] = embedding
    
    return chunks


async def embed_query(query: str) -> List[float]:
    """
    Create embedding for user query.
    
    Args:
        query: User query text
        
    Returns:
        Embedding vector
    """
    return await create_embedding(query)
