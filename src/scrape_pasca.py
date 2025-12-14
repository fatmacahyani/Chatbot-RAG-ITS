import requests
from bs4 import BeautifulSoup, Tag
from pathlib import Path
import json

from sympy import content

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://www.its.ac.id/admission/pascasarjana/informasi-pendaftaran/"


def table_to_text(table: Tag) -> str:
    """Ubah <table> menjadi teks rapi (baris per baris)."""
    rows_text = []
    for row in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
        if cells:
            rows_text.append(" | ".join(cells))
    return "\n".join(rows_text)


def page_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # buang elemen yang tidak perlu
    for tag in soup(["script", "style", "nav", "footer", "header", "svg"]):
        tag.decompose()

    # convert semua tabel ke teks
    for table in soup.find_all("table"):
        txt = table_to_text(table)
        table.replace_with(soup.new_string("\n" + txt + "\n"))

    # ambil teks body
    body = soup.body or soup
    text = body.get_text(separator="\n", strip=True)

    # merapikan baris kosong
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def main():
    print(f"Downloading: {URL}")
    resp = requests.get(URL, timeout=20)
    resp.raise_for_status()

    content = page_to_text(resp.text)

    data = [
        {
            "url": URL,
            "category": "pascasarjana_pendaftaran",
            "description": "Seluruh teks halaman informasi pendaftaran pascasarjana ITS (termasuk tab penting, program, syarat, jadwal, biaya, dll).",
            "content": content,
        }
    ]

    out_path = RAW_DIR / "pasca_raw.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved raw page to:", out_path)
    print("Total chars:", len(content))

    # SIMPAN VERSI TXT
    txt_out = RAW_DIR / "pasca_raw.txt"
    txt_out.write_text(content, encoding="utf-8")

    print("Saved readable text:", txt_out)

if __name__ == "__main__":
    main()
