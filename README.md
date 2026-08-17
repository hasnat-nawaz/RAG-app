# RAG App

A local retrieval-augmented generation (RAG) app. Upload PDFs, ask questions in chat, and get cited answers grounded in your own documents — not the open web.

FastAPI backend, React frontend, Gemini for parsing/embeddings/generation, LanceDB for vectors on disk, and a local cross-encoder for reranking.

## Table of contents

- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Requirements](#requirements)
- [Setup](#setup)
- [Run](#run)
- [API](#api)
- [Local storage](#local-storage)
- [Project layout](#project-layout)
- [Logging](#logging)
- [Notes](#notes)

## What it does

1. **Upload** — Drop a PDF in the chat UI. The backend saves it locally and runs a parallel ingest pipeline.
2. **Ingest** — PDF pages are parsed to markdown, split into chunks, embedded, and stored in LanceDB.
3. **Query** — Ask a question. The app optimizes your query, retrieves relevant chunks (hybrid and/or HyDE), reranks them, and Gemini writes a grounded answer with inline citations and a Sources list.

Everything stays on your machine except Gemini API calls. You can query while a document is still uploading — results reflect whatever chunks are already indexed.

## How it works

### Upload pipeline (3 parallel workers)

When a PDF is uploaded, three async systems run independently, connected by queues:

| Worker | Role |
|--------|------|
| **LLM** | Splits the PDF into 4-page slices, sends them to Gemini in batches of 15 (parallel), waits 65s between batches |
| **Chunker** | Receives each completed LLM batch (all 15 responses together), splits markdown into embeddable chunks |
| **Embedder** | Embeds chunks as they arrive (up to 90 per API call), stores vectors in LanceDB, 65s cooldown between embed rounds |

Each worker has its own retry logic (5 attempts). If ingest fails, partial files and database rows for that source are cleaned up so you can re-upload.

### Query pipeline

1. **Optimize** — Query expansion and query rewriting run in parallel (two Gemini calls)
2. **Retrieve** — As each optimization finishes, retrieval starts immediately (no waiting for the other):
   - Expanded keywords → **BM25** (hybrid)
   - Rewritten query → **semantic search** (cosine similarity on embeddings, hybrid)
   - Rewritten query → **HyDE** (if enabled)
3. **Merge** — Deduplicate chunks by id across methods
4. **Rerank** — Local cross-encoder (`ms-marco-MiniLM-L6-v2`) picks the top-k most relevant chunks
5. **Generate** — Gemini 3.5 Flash writes a cited Markdown answer from the reranked context

## Tech stack

**Backend**

| Layer | Technology |
|-------|------------|
| API | FastAPI, Uvicorn |
| LLM | Google Gemini (`gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-embedding-2`) |
| Vector DB | LanceDB (local, on disk) |
| Chunking | LangChain markdown header splitter |
| Reranking | sentence-transformers cross-encoder (local) |
| PDF parsing | pypdf + Gemini vision |

**Frontend**

| Layer | Technology |
|-------|------------|
| UI | React 19, Vite |
| Rendering | react-markdown |

## Requirements

- Python 3.12+
- Node.js 18+ and npm
- Gemini API key — [Google AI Studio](https://aistudio.google.com/apikey)
- Hugging Face token (optional, recommended) — [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

## Setup

From the project root:

```bash
./setup
```

The script will:

1. Create `backend/.venv` and install Python dependencies
2. Install frontend npm packages
3. Copy `.env.example` → `.env` if you don't have one yet
4. Download and cache the local reranker model

Then edit `.env`:

```env
GEMINI_API_KEY=your_key_here
HF_TOKEN=your_token_here
```

`HF_TOKEN` is only needed for the initial reranker download. After that the model lives in your Hugging Face cache.

## Run

Two terminals:

**Backend**

```bash
source backend/.venv/bin/activate
cd backend
python main.py
```

API: [http://127.0.0.1:8000](http://127.0.0.1:8000)

**Frontend**

```bash
cd frontend
npm run dev
```

| Page | URL |
|------|-----|
| Home | [http://localhost:5173/home](http://localhost:5173/home) |
| Chat | [http://localhost:5173/chat](http://localhost:5173/chat) |

In dev, Vite proxies `/upload`, `/query`, and `/health` to the backend — no separate frontend API URL needed.

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server health check |
| `POST` | `/upload` | Upload a PDF (`multipart/form-data`, field: `file`) |
| `POST` | `/query` | Ask a question (JSON body, see below) |

**Query body**

```json
{
  "query": "What courses are in the 4th semester?",
  "hybrid": true,
  "hyde": false,
  "top_k": 10
}
```

At least one of `hybrid` or `hyde` must be `true`.

**Query response**

```json
{
  "answer": "...",
  "methods": ["hybrid"],
  "documents_retrieved": 15,
  "documents_used": 10
}
```

## Local storage

All persisted data lives under `backend/storage/`. This directory is **gitignored** — it is created automatically on first upload and holds your local corpus.

```text
backend/storage/
├── pdfs/                  # Uploaded PDF files (one file per source name)
├── markdown/              # Reserved for markdown artifacts; cleaned on failed uploads
└── lanceDB/               # LanceDB database (EmbeddingsTable)
    └── EmbeddingsTable.lance/
        ├── data/          # Vector + chunk row files
        ├── _indices/      # FTS (BM25) index
        └── _transactions/ # LanceDB write log
```

| Path | Contents |
|------|----------|
| `pdfs/` | Original uploaded PDFs, keyed by filename |
| `lanceDB/` | Embedded chunks: `id`, `source`, `content`, `metadata`, `vector` |
| `markdown/` | Cleanup target on failed ingest; not populated during normal ingest today |

To wipe everything and start fresh, stop the backend and delete `backend/storage/`.

## Project layout

```text
rag-project/
├── setup                      # One-command install (venv, npm, model cache)
├── README.md
├── .env.example               # Template for API keys (tracked)
├── .env                       # Your keys (gitignored)
│
├── backend/
│   ├── main.py                # FastAPI app entry point
│   ├── bootstrap.py           # sys.path + .env loading
│   ├── server_startup.py      # Lifespan: preload models into app state
│   ├── warmup_models.py       # Reranker download (used by setup)
│   ├── requirements.txt
│   │
│   ├── api/
│   │   ├── routes_upload.py   # POST /upload
│   │   └── routes_query.py    # POST /query
│   │
│   ├── core/
│   │   ├── ingest_pipeline.py     # Parallel LLM → chunker → embedder pipeline
│   │   ├── document_loader.py     # PDF split + Gemini markdown parsing
│   │   ├── chunking.py            # Markdown header-aware text splitter
│   │   ├── embedding.py           # Gemini embedding client
│   │   ├── vector_store.py        # LanceDB read/write, BM25 + semantic + HyDE
│   │   ├── reranker.py            # Local cross-encoder reranking
│   │   ├── generation.py          # Gemini answer generation + citation formatting
│   │   ├── gemini_retry.py        # Shared retry/backoff for API calls
│   │   ├── llm_client.py          # Gemini client singleton + model names
│   │   ├── pipeline_log.py        # One-line tagged logging for upload/query
│   │   └── query_optimization/
│   │       ├── query_expansion.py         # BM25 keyword expansion
│   │       ├── query_rewriting.py         # Semantic search rewrite
│   │       ├── hypothetical_document.py   # HyDE passage generation
│   │       └── common.py                  # Prompt helpers + output cleanup
│   │
│   ├── models/
│   │   └── schemas.py         # Pydantic request/response models
│   │
│   └── storage/               # Local data (gitignored, see above)
│
└── frontend/
    ├── src/
    │   ├── pages/             # HomePage, ChatPage
    │   ├── components/        # Upload, query, results, metrics, method selector
    │   ├── api/client.js      # fetch wrappers for /upload, /query, /health
    │   └── utils/             # Upload time estimate helper
    ├── vite.config.js         # Dev proxy to backend
    └── package.json
```

## Logging

Both pipelines print one-line tagged logs to the backend terminal:

```text
[UPLOAD]   saved report.pdf (2400000 bytes)
[LLM]      batch 1/6 — sending 15 chunks
[CHUNKER]  batch 1 — produced 82 chunks from 15 markdown pieces
[EMBEDDER] round 1 (82 chunks) — embedding
[UPLOAD]   report.pdf — finished in 18m 42s

[QUERY]    started — "What courses are in semester 4?"
[QUERY]    optimizing — expansion + rewrite in parallel
[EXPAND]   ready — "courses semester 4 fourth ..."
[REWRITE]  ready — "What courses are taught in the fourth semester?"
[HYBRID]   BM25 found 8 chunks
[HYBRID]   semantic found 10 chunks
[HYBRID]   merged 15 unique chunks
[HYDE]     hypothetical document — primary prompt
[HYDE]     found 10 chunks
[RERANK]   selected top 10 chunks
[GENERATE] generating answer from 10 chunks
[QUERY]    finished in 12s — 15 retrieved, 10 used
```

Tags: `UPLOAD`, `LLM`, `CHUNKER`, `EMBEDDER`, `QUERY`, `EXPAND`, `REWRITE`, `HYBRID`, `HYDE`, `RERANK`, `GENERATE`.

## Notes

- **PDF only** — Upload accepts `.pdf` files. Duplicate filenames are rejected (409).
- **Empty database** — Queries against an empty store return `"Database is empty."` without an error.
- **API limits** — Gemini quota errors return `"API limit reached. Please try again in a minute."` Other generation failures return a generic unexpected-error message.
- **Free tier** — Upload is quota-sensitive (15 parallel LLM calls per batch, 65s cooldowns). Large PDFs take time; check backend logs for progress.
- **Concurrent use** — Upload and query run independently. You can ask questions while a document is still indexing.
- **Secrets** — Never commit `.env`. Keys stay local.

## Author

[hasnat-nawaz](https://github.com/hasnat-nawaz) · [RAG-app](https://github.com/hasnat-nawaz/RAG-app)
