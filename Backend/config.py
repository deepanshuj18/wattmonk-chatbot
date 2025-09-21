import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "rag_chatbot_index")
    
    # Text Processing Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    
    # API Settings
    API_TITLE: str = "RAG Chatbot API"
    API_DESCRIPTION: str = "Retrieval Augmented Generation Chatbot using FastAPI, Pinecone, and Google Gemini"
    API_VERSION: str = "1.0.0"
    
    # Validation
    def validate_settings(self):
        """Validate that all required settings are provided."""
        missing_keys = []
        if not self.GOOGLE_API_KEY:
            missing_keys.append("GOOGLE_API_KEY")
        if not self.PINECONE_API_KEY:
            missing_keys.append("PINECONE_API_KEY")
        if not self.PINECONE_ENVIRONMENT:
            missing_keys.append("PINECONE_ENVIRONMENT")
            
        if missing_keys:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")

# Create settings instance
settings = Settings()
