from __future__ import annotations

import json
from typing import List, Dict, Any

from app.retrieve.retriever import RetrievedChunk
from app.generate.groq_client import chat_completion


def build_context(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}]\n"
            f"chunk_id: {c.chunk_id}\n"
            f"title: {c.title}\n"
            f"source: {c.source}\n"
            f"text:\n{c.text}\n"
        )
    return "\n\n".join(parts)


def _safe_parse_json(text: str) -> Dict[str, Any]:
    # Groq models sometimes wrap JSON in ```...```
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        # if it says json\n{...}
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return json.loads(t)


def generate_answer_json(question: str, chunks: List[RetrievedChunk]) -> Dict[str, Any]:
    if not chunks:
        return {"answer": "I don’t know based on the provided documents.", "used_sources": []}

    context = build_context(chunks)

    system = (
        "You are an enterprise knowledge copilot.\n"
        "Return ONLY valid JSON (no markdown, no backticks) with this schema:\n"
        "{\n"
        '  "answer": string,\n'
        '  "used_sources": number[]\n'
        "}\n"
        "Rules:\n"
        "1) Use ONLY the provided sources.\n"
        "2) If the sources don’t contain the answer, set answer to exactly:\n"
        "   I don’t know based on the provided documents.\n"
        "   and used_sources to [].\n"
        "3) used_sources must be unique integers referencing Source numbers (1..N).\n"
        "4) Keep the answer concise.\n"
    )

    user = f"Question: {question}\n\nSources:\n{context}"

    raw = chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    data = _safe_parse_json(raw)
    if "answer" not in data or "used_sources" not in data:
        # fallback: be conservative
        return {"answer": "I don’t know based on the provided documents.", "used_sources": []}

    return data


def validate_used_sources(used: List[int], n_sources: int) -> List[int]:
    out = []
    for x in used:
        if isinstance(x, int) and 1 <= x <= n_sources and x not in out:
            out.append(x)
    return out
