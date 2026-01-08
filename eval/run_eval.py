from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"

from app.retrieve.retriever import Retriever  # noqa: E402


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def hit_and_rank(retrieved_doc_ids: List[str], expected: List[str]) -> Tuple[int, int]:
    """
    Returns:
      hit (0/1), rank (1-based rank of first expected doc, else 0)
    """
    exp = set(expected)
    for i, d in enumerate(retrieved_doc_ids, start=1):
        if d in exp:
            return 1, i
    return 0, 0


def mrr_from_rank(rank: int) -> float:
    return 1.0 / rank if rank > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    qs = read_jsonl(QUESTIONS_PATH)
    print(f"Loaded eval questions: {len(qs)} from {QUESTIONS_PATH}")

    retriever = Retriever()

    total = 0
    hits = 0
    mrr_sum = 0.0

    # role breakdown
    by_role: Dict[str, Dict[str, float]] = {}

    for q in qs:
        qid = q["id"]
        role = q.get("role", "public")
        question = q["question"]
        expected = q.get("expected_doc_ids", [])

        chunks = retriever.retrieve(question, top_k=args.k, user_role=role)
        retrieved_doc_ids = [c.metadata.get("doc_id", "") for c in chunks]

        hit, rank = hit_and_rank(retrieved_doc_ids, expected)
        mrr = mrr_from_rank(rank)

        total += 1
        hits += hit
        mrr_sum += mrr

        if role not in by_role:
            by_role[role] = {"total": 0, "hits": 0, "mrr_sum": 0.0}
        by_role[role]["total"] += 1
        by_role[role]["hits"] += hit
        by_role[role]["mrr_sum"] += mrr

        # Print failures (useful for iteration)
        if hit == 0:
            print(f"\nMISS {qid} role={role}")
            print(f"Q: {question}")
            print(f"Expected doc_ids: {expected}")
            print(f"Got doc_ids: {retrieved_doc_ids[:args.k]}")
            if chunks:
                print("Top result preview:")
                print(f"- doc_id={chunks[0].metadata.get('doc_id')} | source={chunks[0].metadata.get('source')}")
                print(f"- snippet={chunks[0].snippet[:160].replace(chr(10),' ')}...")

    hit_rate = hits / total if total else 0.0
    mrr_avg = mrr_sum / total if total else 0.0

    print("\n====================")
    print(f"Retrieval Eval @k={args.k}")
    print(f"Total: {total}")
    print(f"Hit-rate@{args.k}: {hit_rate:.3f} ({hits}/{total})")
    print(f"MRR@{args.k}: {mrr_avg:.3f}")

    print("\n--- By role ---")
    for role, s in by_role.items():
        t = int(s["total"])
        h = int(s["hits"])
        hr = h / t if t else 0.0
        mr = s["mrr_sum"] / t if t else 0.0
        print(f"{role:12s}  hit-rate@{args.k}={hr:.3f} ({h}/{t})  MRR@{args.k}={mr:.3f}")


if __name__ == "__main__":
    main()
