# RAG Chatbot

A Retrieval Augmented Generation (RAG) chatbot built with FastAPI that processes PDF documents, creates embeddings using Google Gemini, stores them in Pinecone, and generates contextual responses.

## Features

- PDF document processing and text extraction
- Text chunking and embedding generation using Google Gemini
- Vector storage in Pinecone for efficient similarity search
- RAG pipeline for contextual response generation
- RESTful API endpoints for chat, PDF upload, and system monitoring
- Error handling, rate limiting, and logging

## Project Structure

```
rag_chatbot/
├── main.py                 # FastAPI app
├── config.py               # Configuration management
├── models/                 # Pydantic models
│   └── schemas.py          # Data models for API
├── services/               # Core business logic
│   ├── pdf_processor.py    # PDF text extraction and chunking
│   ├── embedding_service.py # Gemini embedding generation
│   ├── pinecone_service.py # Vector database operations
│   └── rag_service.py      # RAG pipeline implementation
├── routers/                # API endpoints
│   ├── chat.py             # Chat endpoint
│   ├── pdf.py              # PDF upload endpoint
│   └── system.py           # Health and stats endpoints
├── utils/                  # Helper functions
│   ├── logging_utils.py    # Logging configuration
│   ├── error_handlers.py   # Custom exceptions and handlers
│   └── rate_limiter.py     # API rate limiting
└── requirements.txt        # Dependencies
```

## Prerequisites

- Python 3.8+
- Google Gemini API key
- Pinecone API key and environment

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/rag-chatbot.git
   cd rag-chatbot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with the following variables:
   ```
   GOOGLE_API_KEY=your_gemini_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   PINECONE_INDEX_NAME=rag_chatbot_index
   ```

## Usage

1. Start the FastAPI server:
   ```
   python main.py
   ```
   Or with uvicorn directly:
   ```
   uvicorn main:app --reload
   ```

2. Access the API documentation:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## API Endpoints

### PDF Upload
```
POST /api/upload-pdf
```
Upload a PDF file for processing. The file will be processed, text extracted, chunked, embedded, and stored in Pinecone.

### Chat
```
POST /api/chat
```
Send a message to the chatbot. The message will be processed through the RAG pipeline to generate a contextual response.

### Health Check
```
GET /api/health
```
Check the health of the API and its dependencies (Pinecone, Google AI).

### Index Statistics
```
GET /api/index-stats
```
Get statistics about the Pinecone vector index.

## Development

### Adding New Features

1. Create new service modules in the `services/` directory
2. Add new API endpoints in the `routers/` directory
3. Update the Pydantic models in `models/schemas.py` as needed

### Testing

Run tests with pytest:
```
pytest
```

## Error Handling

The application includes custom exception handling for:
- Rate limiting
- Service unavailability
- Validation errors
- General exceptions

## Performance Considerations

- PDF processing and embedding generation are performed asynchronously
- Batch processing is used for embedding generation and vector storage
- Rate limiting is implemented to prevent API abuse

## License

[MIT License](LICENSE)

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [Google Gemini](https://ai.google.dev/)
- [Pinecone](https://www.pinecone.io/)
- [PyPDF2](https://pypdf2.readthedocs.io/)# wattmonk-chatbot
