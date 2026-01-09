from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from app.retrieve.query_rewrite import rewrite_query
from app.retrieve.rerank import CrossEncoderReranker

# -----------------------
# Runtime knobs (env vars)
# -----------------------
RERANK_ENABLE = os.getenv("RERANK_ENABLE", "0") == "1"
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "30"))
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.2"))
RETRIEVE_CANDIDATES = int(os.getenv("RETRIEVE_CANDIDATES", str(RERANK_CANDIDATES)))


ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION_NAME = "enterprise_kb_copilot"

EMBED_MODEL = "intfloat/e5-small-v2"

ROLE_LEVEL = {
    "public": 0,
    "engineering": 10,
    "security": 20,
    "admin": 30,
}

def role_to_level(role: Optional[str]) -> int:
    return int(ROLE_LEVEL.get((role or "public").lower(), 0))


@dataclass
class RetrievedChunk:
    chunk_id: str
    title: str
    source: str
    snippet: str
    text: str
    metadata: Dict[str, Any]


class Retriever:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.model = SentenceTransformer(EMBED_MODEL)

        # Create reranker only if enabled
        self.reranker: Optional[CrossEncoderReranker] = (
            CrossEncoderReranker() if RERANK_ENABLE else None
        )

    def embed_query(self, q: str) -> List[float]:
        emb = self.model.encode(
            ["query: " + q],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb[0].astype(np.float32).tolist()

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        user_role: str = "public",
    ) -> List[RetrievedChunk]:
        user_level = role_to_level(user_role)

        rewritten = rewrite_query(question)
        q_emb = self.embed_query(rewritten)

        # IMPORTANT:
        # If reranking is ON, fetch a larger candidate pool first, then rerank down to top_k.
        # candidate_k = max(top_k, RERANK_CANDIDATES) if self.reranker else top_k
        candidate_k = max(top_k, RETRIEVE_CANDIDATES)


        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=candidate_k,
            where={"access_level": {"$lte": user_level}},
            include=["documents", "metadatas", "distances"],
        )

        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        candidates: List[RetrievedChunk] = []

        for cid, doc, md, dist in zip(ids, docs, metas, dists):
            if dist is not None and float(dist) > MAX_DISTANCE:
                continue

            title = (md.get("doc_title") if isinstance(md, dict) else "") or ""
            source = (md.get("source") if isinstance(md, dict) else "") or ""
            snippet = (doc[:300] + "...") if doc and len(doc) > 300 else (doc or "")

            meta = dict(md or {})
            meta["vector_distance"] = float(dist) if dist is not None else None

            candidates.append(
                RetrievedChunk(
                    chunk_id=cid,
                    title=title,
                    source=source,
                    snippet=snippet,
                    text=doc or "",
                    metadata=meta,
                )
            )

        # Rerank candidates using cross-encoder
        if self.reranker and candidates:
            scores = self.reranker.score(question, [c.text for c in candidates])
            for c, s in zip(candidates, scores):
                c.metadata["rerank_score"] = float(s)

            candidates.sort(
                key=lambda c: c.metadata.get("rerank_score", -1e9),
                reverse=True,
            )
        
        # After candidates are built and (optionally) reranked/sorted
        # print("\n--- RETRIEVAL DEBUG ---")
        # print(f"question={question!r}")
        # print(f"top_k={top_k}  candidates={len(candidates)}  rerank={'ON' if self.reranker else 'OFF'}")

        # for i, c in enumerate(candidates[:10], start=1):
        #     vd = c.metadata.get("vector_distance")
        #     rs = c.metadata.get("rerank_score")
        #     print(f"{i:02d}. {c.chunk_id}  vec_dist={vd}  rerank_score={rs}  title={c.title}")


        return candidates[:top_k]
