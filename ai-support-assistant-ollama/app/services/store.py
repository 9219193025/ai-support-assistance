import sqlite3, json, os
from typing import List, Tuple
from app.core.config import DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from app.services.embedding import embed_texts

def ensure_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS docs (id INTEGER PRIMARY KEY, source TEXT, text TEXT, embedding TEXT)"
        )
        conn.commit()

def _chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+size]
        chunks.append(chunk)
        i += size - overlap if size - overlap > 0 else size
        if i <= 0:
            break
    return chunks

def add_text(text: str, source: str = "api"):
    ensure_db()
    chunks = _chunk(text)
    embeddings = embed_texts(chunks)
    with sqlite3.connect(DB_PATH) as conn:
        for ch, emb in zip(chunks, embeddings):
            conn.execute(
                "INSERT INTO docs (source, text, embedding) VALUES (?, ?, ?)",
                (source, ch, json.dumps(emb)),
            )
        conn.commit()

def add_chunks(chunks: List[Tuple[str, str]]):
    ensure_db()
    texts = [t for t, _ in chunks]
    embs = embed_texts(texts)
    with sqlite3.connect(DB_PATH) as conn:
        for (text, source), emb in zip(chunks, embs):
            conn.execute(
                "INSERT INTO docs (source, text, embedding) VALUES (?, ?, ?)",
                (source, text, json.dumps(emb)),
            )
        conn.commit()

def fetch_all():
    ensure_db()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT source, text, embedding FROM docs").fetchall()
    out = []
    for source, text, emb in rows:
        out.append({"source": source, "text": text, "embedding": json.loads(emb)})
    return out
