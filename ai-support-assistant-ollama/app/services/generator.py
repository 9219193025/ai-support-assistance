import requests
from app.core.config import OLLAMA_HOST, OLLAMA_CHAT_MODEL

CHAT_URL = f"{OLLAMA_HOST}/api/generate"

def generate_answer(question: str, context: str) -> str:
    system = (
        "You are a helpful support assistant. "
        "Ground your answers STRICTLY in the provided context. "
        "If the context is insufficient, say so and ask for more info."
    )
    prompt = f"{system}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    resp = requests.post(CHAT_URL, json={
        "model": OLLAMA_CHAT_MODEL,
        "prompt": prompt,
        "stream": False
    }, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama generation error {resp.status_code}: {resp.text}")
    data = resp.json()
    if "response" in data:
        return data.get("response","").strip()
    if "message" in data and isinstance(data["message"], dict):
        return data["message"].get("content","").strip()
    return str(data)
