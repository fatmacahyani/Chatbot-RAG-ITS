"""Configuration for ITS Admission RAG Chatbot"""

import os

# Storage paths
DATA_DIR = "data"

# Create data directory
os.makedirs(DATA_DIR, exist_ok=True)

# ITS URLs
ITS_URLS = {
    "link": "https://www.its.ac.id/admission/pascasarjana/informasi-pendaftaran/"
}

# API Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_COLLECTION = "its_admission"

# App Settings  
HOST = "0.0.0.0"
PORT = 8000
