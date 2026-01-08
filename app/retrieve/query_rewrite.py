from __future__ import annotations

from app.generate.groq_client import chat_completion


def rewrite_query(question: str) -> str:
    system = (
        "You rewrite user questions into a short search query for retrieving documentation.\n"
        "Rules:\n"
        "1) Preserve meaning.\n"
        "2) Fix spelling.\n"
        "3) Expand abbreviations (e.g., k8s -> kubernetes).\n"
        "4) Output ONLY the rewritten query string (no quotes, no punctuation commentary).\n"
        "5) Keep it <= 12 words.\n"
    )
    user = f"Question: {question}\nRewrite:"
    out = chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
    )
    return out.strip()
