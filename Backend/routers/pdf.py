from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import Dict, Any
import os
import uuid
import io

from models.schemas import PDFUploadResponse
from services.pdf_processor import process_pdf
from services.embedding_service import embed_chunks
from services.pinecone_service import store_embeddings
from config import settings

router = APIRouter(tags=["pdf"])


async def process_pdf_background(pdf_data_bytes: bytes, filename: str, namespace: str):
    """
    Background task to process PDF, create embeddings, and store in Pinecone.
    
    Args:
        pdf_data_bytes: PDF file data as bytes
        filename: Name of the PDF file
        namespace: Pinecone namespace
    """
    try:
        # Create BytesIO object from raw bytes for PDF processing
        pdf_file_stream = io.BytesIO(pdf_data_bytes)

        # Process PDF
        print(f"Starting PDF processing for {filename}...")
        result = process_pdf(pdf_file_stream, filename)
        print(f"PDF processing complete for {filename}. Pages: {result['pages_processed']}, Chunks: {result['chunks_created']}")
        
        # Create embeddings for chunks
        print(f"Starting embedding creation for {filename}...")
        chunks_with_embeddings = await embed_chunks(result["chunks"])
        print(f"Embedding creation complete for {filename}. Embeddings created: {len(chunks_with_embeddings)}")
        
        # Store embeddings in Pinecone
        print(f"Starting embedding storage in Pinecone for {filename}...")
        await store_embeddings(chunks_with_embeddings, namespace)
        print(f"Successfully stored embeddings for {filename} in Pinecone.")
        
        print(f"Successfully processed PDF: {filename}")
    except Exception as e:
        print(f"Error processing PDF {filename}: {str(e)}")
        # Re-raise the exception to ensure it's logged by the main application logger if possible
        raise


@router.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    namespace: str = "default"
) -> Dict[str, Any]:
    """
    Upload and process a PDF file.
    
    Args:
        background_tasks: FastAPI background tasks
        file: Uploaded PDF file
        namespace: Pinecone namespace
        
    Returns:
        Processing status
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Read file content
        pdf_data = await file.read()
        
        # Generate a unique filename if needed
        filename = file.filename
        
        # Process PDF synchronously to get initial stats
        result = process_pdf(io.BytesIO(pdf_data), filename)
        
        # Schedule background task for embedding and storage
        background_tasks.add_task(
            process_pdf_background,
            pdf_data, # Pass raw bytes to background task
            filename,
            namespace
        )
        
        # Return processing status
        return {
            "filename": filename,
            "pages_processed": result["pages_processed"],
            "chunks_created": result["chunks_created"],
            "status": "processing",
            "message": "PDF uploaded and processing started. Embeddings will be created and stored in the background."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
