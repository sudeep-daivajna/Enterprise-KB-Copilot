from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from app.retrieve.retriever import Retriever
from app.generate.answer import generate_answer_json, validate_used_sources

app = FastAPI(title="Enterprise Knowledge Copilot")

retriever = Retriever()

class AskUser(BaseModel):
    role: str = "public"

class AskRequest(BaseModel):
    question: str
    user: Optional[AskUser] = None

class SourceItem(BaseModel):
    title: str
    source: str
    chunk_id: str
    snippet: str

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    role = (req.user.role if req.user else "public")

    chunks = retriever.retrieve(
        req.question,
        top_k=5,
        user_role=role,
    )

    sources = [
        SourceItem(
            title=c.title,
            source=c.source,
            chunk_id=c.chunk_id,
            snippet=c.snippet,
        )
        for c in chunks
    ]

    gen = generate_answer_json(req.question, chunks)
    used = validate_used_sources(gen.get("used_sources", []), len(sources))

    # Append verified citations 
    answer_text = gen.get("answer", "")
    if used:
        answer_text = answer_text.rstrip() + "\n\nCitations: " + ", ".join([f"[{i}]" for i in used])

    return AskResponse(answer=answer_text, sources=sources)

