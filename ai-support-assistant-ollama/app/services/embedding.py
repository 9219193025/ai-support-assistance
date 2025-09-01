import requests
from typing import List
from app.core.config import OLLAMA_HOST, OLLAMA_EMBED_MODEL

EMBED_URL = f"{OLLAMA_HOST}/api/embeddings"

def embed_texts(texts: List[str]) -> List[List[float]]:
    vectors = []
    for text in texts:
        resp = requests.post(EMBED_URL, json={
            "model": OLLAMA_EMBED_MODEL,
            "input": text
        }, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama embedding error {resp.status_code}: {resp.text}")
        data = resp.json()
        vec = data.get("embedding") or data.get("data", [{}])[0].get("embedding")
        if vec is None:
            raise RuntimeError(f"Invalid embedding response: {data}")
        vectors.append(vec)
    return vectors
