from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


class GroqError(RuntimeError):
    pass


def chat_completion(messages: List[Dict[str, str]], *, temperature: float = 0.2) -> str:
    if not GROQ_API_KEY:
        raise GroqError("Missing GROQ_API_KEY in environment/.env")

    url = f"{GROQ_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 400:
        raise GroqError(f"Groq API error {r.status_code}: {r.text}")

    data = r.json()
    return data["choices"][0]["message"]["content"]
