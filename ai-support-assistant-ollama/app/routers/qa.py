from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List
from app.services.generator import generate_answer
from app.services.search import search_similar
from app.services.store import add_text, ensure_db
from app.ml.naive_bayes import TicketClassifier

router = APIRouter()

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = 5

class AskResponse(BaseModel):
    answer: str
    sources: List[str]

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=5000)

class ClassifyResponse(BaseModel):
    label: str
    probabilities: dict

class IngestRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=200000)

@router.on_event("startup")
def _startup():
    ensure_db()

@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    results = search_similar(req.question, top_k=req.top_k)
    context = "\n\n".join([r["text"] for r in results])
    sources = [r["source"] for r in results]
    answer = generate_answer(req.question, context)
    return AskResponse(answer=answer, sources=sources)

@router.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    clf = TicketClassifier.load_or_train()
    label, probs = clf.predict(req.text)
    return ClassifyResponse(label=label, probabilities=probs)

@router.post("/ingest")
def ingest(req: IngestRequest):
    add_text(req.text, source="api/ingest")
    return {"ok": True}
