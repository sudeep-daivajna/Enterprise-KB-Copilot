from __future__ import annotations

import os
import random
import re
import threading
import time
from typing import Dict, List

from app.generate.groq_client import chat_completion

# GroqError may or may not be exported; handle safely.
try:
    from app.generate.groq_client import GroqError  # type: ignore
except Exception:  # pragma: no cover
    GroqError = Exception  # fallback


# -----------------------
# Knobs (env vars)
# -----------------------
QUERY_REWRITE_ENABLE = os.getenv("QUERY_REWRITE_ENABLE", "1") == "1"

# Groq tier says RPM limit 30; keep a little headroom by default.
GROQ_RPM = int(os.getenv("GROQ_RPM", "25"))
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "8"))

# Backoff settings (used when server doesn't tell us exact wait)
GROQ_RETRY_BASE_SECONDS = float(os.getenv("GROQ_RETRY_BASE_SECONDS", "1.0"))
GROQ_RETRY_MAX_SLEEP_SECONDS = float(os.getenv("GROQ_RETRY_MAX_SLEEP_SECONDS", "30.0"))
GROQ_RETRY_JITTER_SECONDS = float(os.getenv("GROQ_RETRY_JITTER_SECONDS", "0.35"))

# "Please try again in 2s" parser (from your error message)
_TRY_AGAIN_RE = re.compile(r"try again in (\d+)s", re.IGNORECASE)


# -----------------------
# Simple global cache
# -----------------------
_cache: Dict[str, str] = {}


# -----------------------
# Global rate limiter (spacing)
# -----------------------
_min_interval = 60.0 / max(1, GROQ_RPM)  # seconds per request
_lock = threading.Lock()
_next_allowed_ts = 0.0  # monotonic time


def _reserve_call_slot() -> None:
    """
    Enforces ~GROQ_RPM by spacing calls. Thread-safe.
    Reserves a slot first, then sleeps outside the lock.
    """
    global _next_allowed_ts

    wait = 0.0
    with _lock:
        now = time.monotonic()
        if now < _next_allowed_ts:
            wait = _next_allowed_ts - now
            now = _next_allowed_ts  # reserve from the next available slot
        _next_allowed_ts = now + _min_interval

    if wait > 0:
        time.sleep(wait)


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("429" in msg) or ("rate limit" in msg) or ("rate_limit" in msg) or ("rate_limit_exceeded" in msg)


def _sleep_for_rate_limit(e: Exception, attempt: int) -> None:
    """
    If server says 'try again in Xs', honor that.
    Otherwise exponential backoff with jitter.
    """
    msg = str(e)
    m = _TRY_AGAIN_RE.search(msg)
    if m:
        base = float(m.group(1))
        sleep_s = base + 0.15 + random.uniform(0.0, GROQ_RETRY_JITTER_SECONDS)
    else:
        sleep_s = (GROQ_RETRY_BASE_SECONDS * (2 ** attempt)) + random.uniform(0.0, GROQ_RETRY_JITTER_SECONDS)

    time.sleep(min(sleep_s, GROQ_RETRY_MAX_SLEEP_SECONDS))


def _chat_completion_with_throttle(*, messages: List[dict], temperature: float = 0.0) -> str:
    last_exc: Exception | None = None

    for attempt in range(GROQ_MAX_RETRIES + 1):
        # Throttle BEFORE each attempt
        _reserve_call_slot()

        try:
            return chat_completion(messages=messages, temperature=temperature)
        except Exception as e:  # catch wide; groq client may throw GroqError or something else
            last_exc = e
            if _is_rate_limit_error(e):
                _sleep_for_rate_limit(e, attempt)
                continue
            # Not a rate-limit problem => rethrow
            raise

    # Retries exhausted
    assert last_exc is not None
    raise last_exc


def rewrite_query(question: str) -> str:
    """
    Rewrite user question into a short search query.

    Includes:
    - cache
    - RPM throttle
    - 429 retry/backoff
    """
    q = (question or "").strip()
    if not q:
        return ""

    if not QUERY_REWRITE_ENABLE:
        return q

    # Cache hit
    cached = _cache.get(q)
    if cached is not None:
        return cached

    system = (
        "You rewrite user questions into a short search query for retrieving documentation.\n"
        "Rules:\n"
        "1) Preserve meaning.\n"
        "2) Fix spelling.\n"
        "3) Expand abbreviations (e.g., k8s -> kubernetes).\n"
        "4) Output ONLY the rewritten query string (no quotes, no punctuation commentary).\n"
        "5) Keep it <= 12 words.\n"
    )
    user = f"Question: {q}\nRewrite:"

    out = _chat_completion_with_throttle(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
    ).strip()

    # Cache it (even if model returns empty, store it to avoid hammering)
    _cache[q] = out if out else q
    return _cache[q]



# from __future__ import annotations

# from app.generate.groq_client import chat_completion


# def rewrite_query(question: str) -> str:
#     system = (
#         "You rewrite user questions into a short search query for retrieving documentation.\n"
#         "Rules:\n"
#         "1) Preserve meaning.\n"
#         "2) Fix spelling.\n"
#         "3) Expand abbreviations (e.g., k8s -> kubernetes).\n"
#         "4) Output ONLY the rewritten query string (no quotes, no punctuation commentary).\n"
#         "5) Keep it <= 12 words.\n"
#     )
#     user = f"Question: {question}\nRewrite:"
#     out = chat_completion(
#         messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
#         temperature=0.0,
#     )
#     return out.strip()
