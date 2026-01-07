from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from app.retrieve.retiever import Retriever

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

    # Placeholder answer for now (Step 8 will generate with Groq)
    answer = f"Retrieved {len(sources)} sources for role='{role}'. (LLM generation not wired yet.)"

    return AskResponse(answer=answer, sources=sources)
