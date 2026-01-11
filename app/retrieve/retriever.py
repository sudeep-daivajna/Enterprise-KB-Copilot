from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json 
import re
from  rank_bm25 import BM25Okapi

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

BM25_CORPUS_PATH = ROOT / "data" / "bm25" / "bm25_corpus.jsonl"

BM25_ENABLE = os.getenv("BM25_ENABLE", "1") == "1"
BM25_CANDIDATES = int(os.getenv("BM25_CANDIDATES", "30"))

_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)

def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())

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

        self.bm25 = None
        self.bm25_chunk_ids: List[str] = []
        self.bm25_access_levels: List[int] = []

        if BM25_ENABLE and BM25_CORPUS_PATH.exists():
            tokenized_corpus: List[List[str]] = []

            with BM25_CORPUS_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self.bm25_chunk_ids.append(row["chunk_id"])
                    self.bm25_access_levels.append(int(row.get("access_level", 0)))
                    tokenized_corpus.append(row.get("tokens") or [])

            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"✅ BM25 loaded: {BM25_CORPUS_PATH} (chunks={len(self.bm25_chunk_ids)})")
        else:
            print("ℹ️ BM25 disabled or corpus file missing")

      # Create reranker only if enabled
        self.reranker: Optional[CrossEncoderReranker] = (
            CrossEncoderReranker() if RERANK_ENABLE else None
        )

    def bm25_retrieve_ids(self, query: str, user_level: int, k: int) -> List[str]:
        if not self.bm25:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = self.bm25.get_scores(q_tokens)

        # RBAC filter + top-k selection
        idxs = [
            i for i, s in enumerate(scores)
            if self.bm25_access_levels[i] <= user_level and s > 0
        ]
        idxs.sort(key=lambda i: scores[i], reverse=True)

        return [self.bm25_chunk_ids[i] for i in idxs[:k]]


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

        bm25_ids = self.bm25_retrieve_ids(rewritten, user_level, BM25_CANDIDATES)
        bm25_id_set = set(bm25_ids)

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
            meta["from_vector"] = True
            meta["from_bm25"] = cid in bm25_id_set

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

        # Add BM25-only chunks (not already in candidates)
        candidate_id_set = {c.chunk_id for c in candidates}
        missing_bm25_ids = [cid for cid in bm25_ids if cid not in candidate_id_set]

        if missing_bm25_ids:
            fetched = self.collection.get(
                ids=missing_bm25_ids,
                include=["documents", "metadatas"],
            )
            f_ids = fetched.get("ids", [])
            f_docs = fetched.get("documents", [])
            f_metas = fetched.get("metadatas", [])

            for cid, doc, md in zip(f_ids, f_docs, f_metas):
                title = (md.get("doc_title") if isinstance(md, dict) else "") or ""
                source = (md.get("source") if isinstance(md, dict) else "") or ""
                snippet = (doc[:300] + "...") if doc and len(doc) > 300 else (doc or "")

                meta = dict(md or {})
                meta["vector_distance"] = None
                meta["from_vector"] = False
                meta["from_bm25"] = True

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
        print("\n--- RETRIEVAL DEBUG ---")
        print(f"question={question!r}")
        print(f"top_k={top_k}  candidates={len(candidates)}  rerank={'ON' if self.reranker else 'OFF'}")

        for i, c in enumerate(candidates[:10], start=1):
            vd = c.metadata.get("vector_distance")
            rs = c.metadata.get("rerank_score")
            fv = c.metadata.get("from_vector")
            fb = c.metadata.get("from_bm25")
            print(f"{i:02d}. {c.chunk_id} vec_dist={vd} rerank={rs} v={fv} b={fb} title={c.title}")

        v_only = [c for c in candidates if c.metadata.get("from_vector") and not c.metadata.get("from_bm25")]
        b_only = [c for c in candidates if c.metadata.get("from_bm25") and not c.metadata.get("from_vector")]
        both   = [c for c in candidates if c.metadata.get("from_vector") and c.metadata.get("from_bm25")]

        print(f"counts: v_only={len(v_only)} b_only={len(b_only)} both={len(both)} total={len(candidates)}")

        print("\nBM25-only (first 10):")
        for c in b_only[:10]:
            print(f"- {c.chunk_id} title={c.title}")


        return candidates[:top_k]
