import os
import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
import numpy as np
from datetime import datetime
import argparse

@dataclass
class Document:
    """Represents a document chunk with metadata"""
    id: str
    content: str
    metadata: Dict
    embedding: Optional[List[float]] = None

class RAGChatbot:
    def __init__(self, 
                 collection_name: str = "gemini_documents",
                 google_api_key: Optional[str] = None,
                 embedding_model_name: str = "models/embedding-001",
                 persist_directory: str = "./chroma_db_gemini"):
        """
        Initialize the RAG Chatbot using Google Gemini
        
        Args:
            collection_name: Name of the ChromaDB collection
            google_api_key: Google AI API key
            embedding_model_name: Google's model for embeddings
            persist_directory: Directory to persist ChromaDB data
        """
        if not google_api_key:
            raise ValueError("Google AI API key is required.")
        
        genai.configure(api_key=google_api_key)
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        
        # Initialize LLM for generation
        self.llm = genai.GenerativeModel('gemini-2.5-flash')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )
        print(f"Loaded/Created collection: {collection_name}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a given text."""
        if not text.strip():
            print("Attempted to embed an empty string. Returning zero vector.")
            return [0.0] * 768  # Standard size for embedding-001
        try:
            return genai.embed_content(
                model=self.embedding_model_name,
                content=text,
                task_type="retrieval_document"
            )["embedding"]
        except Exception as e:
            print(f"Error generating embedding for text: '{text[:50]}...'\nError: {e}")
            return [0.0] * 768

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Splits text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks
    
    def add_document(self, content: str, metadata: Dict = None) -> str:
        """Adds a document to the knowledge base."""
        if metadata is None: metadata = {}
        
        doc_id = hashlib.md5(content.encode()).hexdigest()
        metadata.update({
            'timestamp': datetime.now().isoformat(),
            'length': len(content)
        })
        
        chunks = self.chunk_text(content)
        chunk_ids = [f"{doc_id}_{i}" for i, _ in enumerate(chunks)]
        chunk_metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'parent_doc_id': doc_id
            })
            chunk_metadatas.append(chunk_metadata)
        
        # Generate embeddings for all chunks
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = [self._get_embedding(chunk) for chunk in chunks]
        
        # Add to ChromaDB
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=chunk_ids,
            metadatas=chunk_metadatas
        )
        print(f"Added document {doc_id} with {len(chunks)} chunks.")
        return doc_id

    def retrieve_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """Retrieves relevant documents for a query."""
        query_embedding = self._get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        retrieved_docs = []
        if results.get('documents'):
            for i, doc in enumerate(results['documents'][0]):
                retrieved_docs.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'id': results['ids'][0][i]
                })
        return retrieved_docs
    
    def generate_response(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Generates a response using retrieved documents."""
        if not retrieved_docs:
            return "I couldn't find any relevant information to answer your question."

        context = "\n\n".join([
            f"Document {i+1} (Source: {doc['metadata'].get('source_file', 'N/A')}): {doc['content']}"
            for i, doc in enumerate(retrieved_docs[:3])
        ])
        
        prompt = f"""You are a helpful assistant. Answer the following question based ONLY on the context provided below. If the information is not in the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            response = self.llm.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return "There was an error generating a response."

    def chat(self, query: str) -> Dict:
        """Main chat function."""
        retrieved_docs = self.retrieve_documents(query)
        response = self.generate_response(query, retrieved_docs)
        
        return {
            'query': query,
            'response': response,
            'retrieved_docs': retrieved_docs
        }

    def get_collection_stats(self) -> Dict:
        """Get statistics about the document collection."""
        return {'total_chunks': self.collection.count()}

def main():
    """Main function for interactive command-line chat."""
    parser = argparse.ArgumentParser(description="RAG Chatbot with Gemini and ChromaDB")
    parser.add_argument("--google-key", required=True, help="Google AI API key")
    args = parser.parse_args()
    
    chatbot = RAGChatbot(google_api_key=args.google_key)
    
    print("Gemini RAG Chatbot initialized!")
    print("Type your question or '/quit' to exit.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input: continue
            if user_input.lower() == "/quit": break
            
            result = chatbot.chat(user_input)
            print(f"\nBot: {result['response']}")
            
            if result['retrieved_docs']:
                print(f"\n(Based on {len(result['retrieved_docs'])} retrieved document chunks)")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()