from pathlib import Path
import os
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY belum di-set di .env")

# Model & parameter LLM
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "350"))

EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

client = OpenAI(api_key=OPENAI_API_KEY)

# Init Chroma & Embedding model
_chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _chroma_client.get_or_create_collection(name="its_pasca")
_embed_model = SentenceTransformer(EMBED_MODEL_NAME)


def retrieve_context(query: str, k: int = 5):
    """
    Similarity search di ChromaDB.
    Untuk query biaya/UKT, lakukan query lebih luas dan prioritaskan chunk yang mengandung 'Rp'.
    """
    q_lower = query.lower()

    # Query khusus biaya
    if any(kw in q_lower for kw in ["biaya", "ukt", "uang kuliah", "spp"]):
        expanded_query = query + " biaya kuliah Rp rupiah per semester pascasarjana ITS"
    else:
        expanded_query = query

    q_emb = _embed_model.encode([expanded_query], convert_to_numpy=True)[0].tolist()

    results = _collection.query(
        query_embeddings=[q_emb],
        n_results=12,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    pairs = list(zip(documents, metadatas))

    # Jika query tentang biaya, prioritaskan dokumen yang mengandung 'Rp'
    if any(kw in q_lower for kw in ["biaya", "ukt", "uang kuliah", "spp"]):
        biaya_pairs = [p for p in pairs if "rp" in (p[0] or "").lower()]
        return (biaya_pairs[:k] if biaya_pairs else pairs[:k])

    return pairs[:k]


def build_system_prompt(mode: str = "rag") -> str:
    """
    mode:
      - 'rag'     : jawab berdasarkan konteks
      - 'non_rag' : baseline LLM tanpa konteks dokumen
    """
    common = (
        "Kamu adalah asisten informasi pendaftaran Pascasarjana ITS.\n"
        "Gunakan bahasa Indonesia yang baku, ringkas, dan langsung ke inti.\n"
        "\n"
        "Aturan gaya jawab:\n"
        "- Jawab HANYA hal yang eksplisit ditanyakan pengguna.\n"
        "- Jika pengguna menanyakan satu syarat spesifik (mis. skor TOEFL saja), jawab hanya syarat itu.\n"
        "- Jika pertanyaan meminta daftar/komponen, gunakan bullet/numbering.\n"
        "- Jangan menambahkan syarat/konteks lain yang tidak diminta.\n"
        "\n"
        "Contoh:\n"
        "User: 'berapa syarat toefl program doktor?'\n"
        "Jawab: 'Skor TOEFL minimal untuk Program Doktor ITS adalah 500.'\n"
    )

    if mode == "non_rag":
        return (
            common
            + "\n"
            "PENTING (baseline non-RAG):\n"
            "- Kamu TIDAK memiliki akses dokumen resmi ITS, kamu bisa menebak atau memberikan jawaban berdasarkan pengetahuan umum.\n"
            "- Jika tidak yakin, jawab:\n"
            "\"Maaf, informasi tersebut tidak diketahui.\""
        )

    # mode rag
    return (
        common
        + "\n"
        "PENTING (RAG):\n"
        "- Gunakan HANYA informasi yang ada di KONTEN/KONTEKS.\n"
        "- Jika pertanyaan tentang BIAYA/UKT, cari angka yang diawali 'Rp' di konteks.\n"
        "- Jika informasi tidak ada di konteks, jawab persis:\n"
        "\"Maaf, informasi tersebut tidak tersedia dalam dokumen resmi yang saya miliki.\""
    )


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Wrapper pemanggilan LLM agar parameter konsisten untuk RAG & Non-RAG.
    """
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()


def generate_answer_rag(query: str, context_items):
    """
    Panggil LLM dengan konteks RAG.
    """
    if not context_items:
        context_text = "(Tidak ada konteks yang ditemukan.)"
    else:
        parts = []
        for i, (ctx, meta) in enumerate(context_items, start=1):
            url = meta.get("url", "")
            category = meta.get("category", "")
            parts.append(f"[Sumber {i} - {category} - {url}]\n{ctx}")
        context_text = "\n\n".join(parts)

    user_prompt = (
        f"Pertanyaan pengguna:\n{query}\n\n"
        f"Konteks dari dokumen resmi ITS:\n{context_text}\n\n"
        "Instruksi:\n"
        "- Jawab singkat (1–4 kalimat). Jika perlu daftar, gunakan bullet.\n"
        "- Jika tidak ada data relevan di konteks, gunakan kalimat penolakan yang sudah ditentukan.\n"
    )

    return _call_llm(build_system_prompt(mode="rag"), user_prompt)


def generate_answer_nonrag(query: str):
    """
    Baseline LLM tanpa retrieval konteks.
    """
    user_prompt = (
        f"Pertanyaan pengguna:\n{query}\n\n"
        "Instruksi:\n"
        "- Jawab singkat (1–2 kalimat) dan hanya sesuai pertanyaan.\n"
        "- Jika tidak yakin/ tidak punya info pasti, gunakan kalimat penolakan yang sudah ditentukan.\n"
    )
    return _call_llm(build_system_prompt(mode="non_rag"), user_prompt)


def rag_answer(query: str):
    """
    Input query → output jawaban + sumber (RAG).
    """
    context_items = retrieve_context(query, k=5)
    answer = generate_answer_rag(query, context_items)

    sources = []
    for _, meta in context_items:
        sources.append(
            {
                "url": meta.get("url", ""),
                "category": meta.get("category", ""),
                "description": meta.get("description", ""),
            }
        )

    return {
        "answer": answer,
        "sources": sources,
        "mode": "rag",
    }


def nonrag_answer(query: str):
    """
    Input query → output jawaban baseline tanpa sumber (Non-RAG).
    """
    answer = generate_answer_nonrag(query)
    return {
        "answer": answer,
        "sources": [],
        "mode": "non_rag",
    }
