# Updated add_data.py

import argparse
import os
import fitz  # The PyMuPDF library
from gemini_rag_chatbot import RAGChatbot # Import the class from the other file

def ingest_file(chatbot: RAGChatbot, file_path: str):
    """
    Detects file type and adds its content to the knowledge base.
    Supports .txt and .pdf files.
    
    Args:
        chatbot: An instance of the RAGChatbot
        file_path: Path to the file to ingest
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # --- PDF File Handling ---
    if file_path.lower().endswith('.pdf'):
        print(f"Processing PDF file: {file_path}")
        try:
            doc = fitz.open(file_path)
            full_text = ""
            for page_num, page in enumerate(doc):
                full_text += page.get_text()
            doc.close()
            
            if full_text.strip():
                print(f"Extracted {len(full_text)} characters from PDF.")
                metadata = {'source_file': file_path}
                # Add the entire PDF content as a single document
                chatbot.add_document(full_text, metadata)
                print(f"Successfully added content from {file_path}")
            else:
                print(f"Warning: No text could be extracted from {file_path}.")

        except Exception as e:
            print(f"An error occurred while processing the PDF: {e}")

    # --- Text File Handling ---
    elif file_path.lower().endswith('.txt'):
        print(f"Processing text file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                doc_count = 0
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        metadata = {
                            'source_file': file_path,
                            'line_number': line_num
                        }
                        chatbot.add_document(line, metadata)
                        doc_count += 1
                print(f"Successfully added {doc_count} documents from {file_path}")
        except Exception as e:
            print(f"An error occurred while reading the text file: {e}")

    else:
        print(f"Error: Unsupported file type '{os.path.basename(file_path)}'. Only .txt and .pdf are supported.")


def main():
    """Main function to handle adding data."""
    parser = argparse.ArgumentParser(description="Add data to the Gemini RAG Chatbot's knowledge base from a file (.txt or .pdf).")
    parser.add_argument("--file", required=True, help="Path to the file to be added.")
    parser.add_argument("--google-key", required=True, help="Your Google AI API key.")
    
    args = parser.parse_args()
    
    # Initialize the chatbot to use its data handling methods
    print("Initializing chatbot to add data...")
    chatbot = RAGChatbot(google_api_key=args.google_key)
    
    # Ingest the specified file
    ingest_file(chatbot, args.file)
    
    # Print final stats
    stats = chatbot.get_collection_stats()
    print(f"\nDatabase now contains {stats['total_chunks']} text chunks.")

if __name__ == "__main__":
    main()