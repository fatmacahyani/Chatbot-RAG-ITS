from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "pasca_raw.json"
PROC_DIR = BASE_DIR / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 500  # karakter per chunk
OVERLAP = 100     # overlap antar chunk


def chunk_text(text: str, chunk_size: int, overlap: int):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def main():
    data = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    processed = []
    idx = 0

    for page in data:
        content = page["content"]
        chunks = chunk_text(content, CHUNK_SIZE, OVERLAP)
        for ch in chunks:
            processed.append(
                {
                    "id": f"doc_{idx}",
                    "url": page["url"],
                    "category": page["category"],
                    "description": page.get("description", ""),
                    "text": ch,
                }
            )
            idx += 1

    out_path = PROC_DIR / "pasca_chunks.json"
    out_path.write_text(
        json.dumps(processed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved chunks to {out_path} (total {len(processed)} chunks)")


if __name__ == "__main__":
    main()
