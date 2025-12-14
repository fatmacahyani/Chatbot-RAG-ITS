from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag_pipeline import rag_answer, nonrag_answer

app = FastAPI(
    title="Chatbot Pendaftaran Pascasarjana ITS",
    description="RAG chatbot untuk informasi pendaftaran pascasarjana ITS.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    mode: str | None = "rag"   # "rag" atau "non_rag"

@app.post("/chat")
def chat(req: ChatRequest):
    mode = (req.mode or "rag").lower()
    if mode in ["non_rag", "nonrag", "baseline"]:
        return nonrag_answer(req.query)
    return rag_answer(req.query)

# endpoint khusus non-rag
@app.post("/chat_nonrag")
def chat_nonrag(req: ChatRequest):
    return nonrag_answer(req.query)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )