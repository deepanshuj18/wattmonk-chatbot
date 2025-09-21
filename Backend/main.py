# main_api.py

import os
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# --- New lines to load the .env file ---
from dotenv import load_dotenv
load_dotenv()  # This reads the .env file and loads the variables into the environment

# --- Core Chatbot Logic ---
# We import the RAGChatbot class from your existing file
from gemini_rag_chatbot import RAGChatbot

# --- Environment Variable for API Key ---
# This part now works because load_dotenv() has already populated the environment
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Make sure it's set in your .env file.")

# --- Initialize Chatbot ---
# This creates a single, shared instance of the chatbot for the entire API
print("Initializing RAG Chatbot...")
chatbot = RAGChatbot(google_api_key=api_key)
print("Chatbot initialized successfully.")

# --- FastAPI App ---
app = FastAPI(
    title="Gemini RAG Chatbot API",
    description="An API for interacting with a Retrieval-Augmented Generation chatbot powered by Google Gemini.",
    version="1.0.0",
)
# --- CORS Configuration ---
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request and Response Data ---
class AddTextRequest(BaseModel):
    content: str = Field(..., example="Project Nautilus is a high-priority initiative.")
    metadata: Optional[Dict] = Field(None, example={"source": "manual_input"})

class ChatRequest(BaseModel):
    query: str = Field(..., example="What is Project Nautilus?")

class DocumentMetadata(BaseModel):
    source_file: Optional[str] = None
    line_number: Optional[int] = None
    chunk_index: int
    total_chunks: int
    # Add other metadata fields you expect to see

class RetrievedDoc(BaseModel):
    content: str
    metadata: Dict # Kept as Dict for flexibility as metadata can vary
    distance: float
    id: str

class ChatResponse(BaseModel):
    query: str
    response: str
    retrieved_docs: List[RetrievedDoc]


# --- API Endpoints ---

@app.get("/", tags=["General"])
async def read_root():
    """A simple health check endpoint."""
    return {"status": "API is running"}

@app.post("/add-text", tags=["Knowledge Base"], status_code=201)
async def add_text(request: AddTextRequest):
    """
    Adds a piece of text to the knowledge base.
    """
    try:
        doc_id = chatbot.add_document(request.content, request.metadata)
        return {"message": "Text added successfully", "document_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.post("/add-file", tags=["Knowledge Base"], status_code=201)
async def add_file(file: UploadFile = File(...)):
    """
    Adds content from an uploaded file (.txt or .pdf) to the knowledge base.
    """
    if not (file.filename.lower().endswith('.pdf') or file.filename.lower().endswith('.txt')):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a .txt or .pdf file.")
    
    try:
        file_bytes = await file.read()
        
        if file.filename.lower().endswith('.pdf'):
            # Process PDF from in-memory bytes
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            full_text = "".join(page.get_text() for page in doc)
            doc.close()
            content_source = full_text
        else: # .txt file
            content_source = file_bytes.decode('utf-8')
        
        if not content_source.strip():
            raise HTTPException(status_code=400, detail=f"No text content found in file '{file.filename}'.")
            
        doc_id = chatbot.add_document(content_source, metadata={'source_file': file.filename})
        return {"message": f"File '{file.filename}' processed successfully.", "document_id": doc_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_with_bot(request: ChatRequest):
    """
    Sends a query to the chatbot and gets a response.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        result = chatbot.chat(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during chat processing: {e}")


@app.get("/stats", tags=["Knowledge Base"])
async def get_stats():
    """
    Gets statistics about the knowledge base.
    """
    return chatbot.get_collection_stats()