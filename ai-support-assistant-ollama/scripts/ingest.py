import argparse, os
from app.services.store import ensure_db, add_chunks

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_file(path: str, chunk_size=1500, overlap=200):
    text = read_text_file(path)
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
        i += chunk_size - overlap if chunk_size - overlap > 0 else chunk_size
        if i <= 0:
            break
    return [(c, os.path.basename(path)) for c in chunks]

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest text-like files into the RAG store.")
    parser.add_argument("--path", required=True, help="Folder containing .txt/.md/.log etc.")
    args = parser.parse_args()

    ensure_db()
    root = args.path
    all_chunks = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".txt", ".md", ".log")):
                p = os.path.join(dirpath, fn)
                all_chunks.extend(chunk_file(p))

    if not all_chunks:
        print("No text-like files found. Add .txt/.md/.log files into the folder.")
        return

    add_chunks(all_chunks)
    print(f"Ingested {len(all_chunks)} chunks into {os.getenv('DB_PATH','rag_store.db')}")

if __name__ == "__main__":
    main()
