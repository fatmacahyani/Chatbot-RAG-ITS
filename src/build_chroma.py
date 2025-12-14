from pathlib import Path
import json
import chromadb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_PATH = BASE_DIR / "data" / "processed" / "pasca_chunks.json"
CHROMA_DIR = BASE_DIR / "chroma_db"

load_dotenv()
EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="its_pasca")

    if collection.count() > 0:
        collection.delete(where={}) 

    model = SentenceTransformer(EMBED_MODEL_NAME)

    data = json.loads(PROC_PATH.read_text(encoding="utf-8"))

    ids = []
    documents = []
    metadatas = []

    for row in data:
        ids.append(row["id"])
        documents.append(row["text"])
        metadatas.append(
            {
                "url": row["url"],
                "category": row["category"],
                "description": row.get("description", ""),
            }
        )

    print(f"Encoding {len(documents)} chunks...")
    embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.tolist()

    print("Adding to Chroma...")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    print("Done. ChromaDB stored at", CHROMA_DIR)


if __name__ == "__main__":
    main()
