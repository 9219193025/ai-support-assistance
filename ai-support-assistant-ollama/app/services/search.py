from typing import List, Dict
import math
from app.services.store import fetch_all
from app.services.embedding import embed_texts

def _cosine(a: List[float], b: List[float]) -> float:
    num = 0.0
    da = 0.0
    db = 0.0
    for x, y in zip(a, b):
        num += x * y
        da += x * x
        db += y * y
    if da == 0 or db == 0:
        return 0.0
    return num / (math.sqrt(da) * math.sqrt(db))

def search_similar(query: str, top_k: int = 5) -> List[Dict]:
    items = fetch_all()
    if not items:
        return []
    q_emb = embed_texts([query])[0]
    scored = []
    for it in items:
        sim = _cosine(q_emb, it["embedding"])
        scored.append({"source": it["source"], "text": it["text"], "score": sim})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
