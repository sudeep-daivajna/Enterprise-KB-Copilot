from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from app.retrieve.query_rewrite import rewrite_query

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

        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where={"access_level": {"$lte": user_level}},
            include=["documents", "metadatas", "distances"],
        )

        out: List[RetrievedChunk] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        MAX_DISTANCE = 0.2

        for cid, doc, md, dist in zip(ids, docs, metas, dists):
            if dist is not None and float(dist) > MAX_DISTANCE:
                continue

            title = (md.get("doc_title") if isinstance(md, dict) else "") or ""
            source = (md.get("source") if isinstance(md, dict) else "") or ""
            snippet = (doc[:300] + "...") if doc and len(doc) > 300 else (doc or "")
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    title=title,
                    source=source,
                    snippet=snippet,
                    text=doc or "",
                    metadata=md or {},
                )
            )

        return out
