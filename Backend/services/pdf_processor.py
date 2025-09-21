import os
import re
import uuid
from typing import List, Dict, Any, BinaryIO
from PyPDF2 import PdfReader
from config import settings


def extract_text_from_pdf(pdf_file: BinaryIO, filename: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF pages.
    
    Args:
        pdf_file: File-like object containing PDF data
        filename: Name of the PDF file
        
    Returns:
        List of dictionaries with page text and metadata
    """
    try:
        reader = PdfReader(pdf_file)
        pages = []
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():  # Only add non-empty pages
                pages.append({
                    "text": text,
                    "page_number": i + 1,
                    "source": filename
                })
        
        if not pages:
            print(f"WARNING: No text extracted from PDF '{filename}'. It might be an image-based PDF or corrupted.")
                
        return pages
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF: {str(e)}")


def clean_text(text: str) -> str:
    """
    Clean and preprocess text data.
    
    Args:
        text: Raw text from PDF
        
    Returns:
        Cleaned text
    """
    # Replace multiple newlines with single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove any non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    
    return text.strip()


def chunk_text(text: str, page_number: int, source: str, 
               chunk_size: int = None, chunk_overlap: int = None) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to split
        page_number: Page number in the source document
        source: Source document name
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of dictionaries with chunk text and metadata
    """
    # Use settings if not explicitly provided
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    
    # Clean the text
    text = clean_text(text)
    
    # If text is shorter than chunk size, return as single chunk
    if len(text) <= chunk_size:
        return [{
            "text": text,
            "page_number": page_number,
            "source": source,
            "chunk_id": str(uuid.uuid4())
        }]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Get chunk of text
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to find a good breaking point (end of sentence or paragraph)
        if end < len(text):
            # Look for paragraph break
            paragraph_break = chunk.rfind('\n')
            # Look for sentence break (period followed by space)
            sentence_break = chunk.rfind('. ')
            
            # Use the latest good breaking point
            if paragraph_break > chunk_size * 0.5:
                end = start + paragraph_break + 1
                chunk = text[start:end]
            elif sentence_break > chunk_size * 0.5:
                end = start + sentence_break + 2  # Include the period and space
                chunk = text[start:end]
        
        # Add chunk with metadata
        chunks.append({
            "text": chunk.strip(),
            "page_number": page_number,
            "source": source,
            "chunk_id": str(uuid.uuid4())
        })
        
        # Move start position for next chunk, considering overlap
        start = end - chunk_overlap
        
    return chunks


def process_pdf(pdf_file: BinaryIO, filename: str) -> Dict[str, Any]:
    """
    Process PDF file: extract text, clean, and chunk.
    
    Args:
        pdf_file: File-like object containing PDF data
        filename: Name of the PDF file
        
    Returns:
        Dictionary with processing results
    """
    # Extract text from PDF pages
    pages = extract_text_from_pdf(pdf_file, filename)
    
    all_chunks = []
    
    # Process each page
    for page in pages:
        page_chunks = chunk_text(
            text=page["text"],
            page_number=page["page_number"],
            source=page["source"]
        )
        all_chunks.extend(page_chunks)
    
    return {
        "filename": filename,
        "pages_processed": len(pages),
        "chunks_created": len(all_chunks),
        "chunks": all_chunks
    }
