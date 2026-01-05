from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Enterprise Knowledge Copilot")

class AskUser(BaseModel):
    role : str = "public"

class AskRequest(BaseModel):
    question : str
    user : Optional[AskUser] = None

class SourceItem(BaseModel):
    title : str
    source : str
    chunk_id : str
    snippet : str

class AskResponse(BaseModel):
    answer : str
    sources: List[SourceItem]

@app.get("/health")
def health():
    return {"status" : "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req : AskRequest):
    role = (req.user.role if req.user else "public")

    return AskResponse(
        answer=f"Stub answer. question='{req.question}' role='{role}'",
        sources=[]
    )