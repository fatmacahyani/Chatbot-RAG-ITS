# Chatbot RAG ITS

A Retrieval-Augmented Generation (RAG) chatbot for ITS (Institut Teknologi Sepuluh Nopember) information.

## Features

- Web scraping of ITS data
- Document processing and chunking
- Vector database with ChromaDB
- RAG pipeline with OpenAI
- Web interface for chatbot interaction
- Evaluation notebooks for testing

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fatmacahyani/Chatbot-RAG-ITS.git
   cd Chatbot-RAG-ITS
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys:
   # OPENAI_API_KEY=your_openai_api_key_here
   # HF_TOKEN=your_huggingface_token_here
   ```

4. **Run the application:**
   ```bash
   uvicorn main:app --reload
   ```

## API Endpoints

- `GET /` - Web interface
- `POST /chat` - Chat with the bot
- `POST /build-database` - Build the vector database

## Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key
- `HF_TOKEN` - Your Hugging Face token
