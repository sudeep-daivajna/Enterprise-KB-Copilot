# Enterprise Knowledge Copilot (RAG) — Groq + HuggingFace + Chroma

A lightweight “enterprise-style” knowledge copilot that answers questions over technical documentation **with citations** and **role-based access control (RBAC)**.

**Corpus (demo):**
- AWS Well-Architected Framework (PDF)
- Kubernetes Concepts docs (HTML)

## Features
- ✅ Ingestion: download + parse HTML/PDF into a unified document format
- ✅ Chunking: heading-aware + token-ish chunking with stable `chunk_id`s
- ✅ Vector search: Chroma + HuggingFace embeddings (`intfloat/e5-small-v2`)
- ✅ Grounded answers: Groq LLM generates answers using retrieved context only
- ✅ Citations you can trust: model outputs structured `used_sources`, API validates + returns only used chunks
- ✅ RBAC: `public` vs `engineering` access enforced at retrieval time
- ✅ Evaluation: retrieval Hit-rate@k + MRR@k with an eval harness

## Architecture

Offline indexing:
`docs -> parse -> chunk -> embed -> store (Chroma + metadata)`

Online query:
`question -> (query rewrite) -> retrieve (RBAC filter + relevance gate) -> generate -> answer + citations`

## Quickstart

### 1) Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

Create `.env`:
```env
GROQ_API_KEY=...
# optional:
# GROQ_MODEL=llama-3.3-70b-versatile
```

### 2) Build the corpus + index
```bash
python app/ingest/download_corpus.py
python app/ingest/parse_docs.py
python app/chunk/make_chunks.py
python app/index/build_index.py --reset
```

### 3) Run the API
```bash
uvicorn app.api.main:app --reload --port 8000
```

### 4) Ask a question
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"What are the pillars of the AWS Well-Architected Framework?\",\"user\":{\"role\":\"public\"}}"
```

## RBAC Demo
- `public` **cannot** access Kubernetes docs
- `engineering` can

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"What is a Kubernetes Service?\",\"user\":{\"role\":\"public\"}}"
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"What is a Kubernetes Service?\",\"user\":{\"role\":\"engineering\"}}"
```

## Evaluation
Run retrieval evaluation:
```bash
python eval/run_eval.py --k 5
```

Example results (fill in from your latest run):
- Hit-rate@5: **1.00**
- MRR@5: **0.927**

Ablation:
- Adding **LLM query rewrite** improved MRR@5 from **0.892 → 0.927** on a noisy/typo-augmented set.

## Notes / Design choices
- **Stable chunk IDs** enable debugging + evaluation + future feedback loops.
- **Relevance gating** prevents out-of-domain nearest-neighbor noise.
- **Structured citations** prevent “citation hallucinations” and ensure returned sources match claims.
