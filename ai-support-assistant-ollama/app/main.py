from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.qa import router as qa_router
from pydantic import BaseModel


# Create FastAPI instance
app = FastAPI(title="AI Support Assistant (Ollama)")
@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI project!"}

# Allow CORS for frontend (adjust origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with ["http://localhost:3000"] if you want only frontend access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check route
@app.get("/health")
def health():
    return {"status": "ok"}
class TextInput(BaseModel):
    text: str

@app.post("/summarize")
def summarize(input: TextInput):
    # Simple placeholder logic (replace with AI model)
    words = input.text.split()
    summary = " ".join(words[:min(20, len(words))]) + "..."
    return {"summary": summary}
with open("data/mydocs.txt", "r", encoding="utf-8") as f:
    docs = f.read()

print("=== File Content Loaded ===")
print(docs[:200])  # print first 200 characters only
print("=== End of Preview ===")
@app.get("/read-file")
def read_file():
    with open("data/mydocs.txt", "r", encoding="utf-8") as f:
        docs = f.read()
    return {"content": docs[:500]} 
 # send first 500 characters



# Include QA router
app.include_router(qa_router, prefix="/qa", tags=["Question-Answer"])
