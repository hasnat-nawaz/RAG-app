# RAG App

A local retrieval-augmented generation (RAG) app. Upload PDFs, ask questions in chat, and get answers grounded in your own documents — not the open web.

Built as a small full-stack project: FastAPI on the backend, React on the frontend, with embeddings and vectors stored on your machine.

## Table of contents

- [What it does](#what-it-does)
- [Tech stack](#tech-stack)
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Run](#run)
- [Project layout](#project-layout)
- [Notes](#notes)

## What it does

You add PDF sources through the UI. The backend turns them into markdown, splits them into chunks, embeds those chunks, and saves them in a local LanceDB store.

When you ask a question, the app retrieves the most relevant chunks (dense search, BM25, hybrid, and optional HyDE), reranks them, then asks Gemini to answer using that context.

Everything stays on your machine except the Gemini API calls used for parsing, embeddings, and generation.

## Tech stack

**Backend**

- Python 3.12+
- FastAPI + Uvicorn
- Google Gemini (document parsing, embeddings, answers)
- LanceDB (local vector store)
- LangChain text splitters
- sentence-transformers (reranking)

**Frontend**

- React 19
- Vite
- react-markdown

## Features

- Upload PDFs from the chat UI (drag-and-drop or file picker)
- Local ingest pipeline: PDF → markdown → chunks → embeddings → LanceDB
- Chat over your uploaded sources
- Retrieval options: hybrid search and HyDE
- Live server status in the UI
- One-command dependency setup via `./setup`

## Requirements

- Python 3
- Node.js 18+ and npm
- A Gemini API key ([Google AI Studio](https://aistudio.google.com/apikey))
- A Hugging Face access token (optional but recommended) — create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)


## Setup

Clone the repo, then from the project root:

```bash
./setup
```

That script will:

1. Create `backend/.venv` if it does not exist
2. Install Python packages from `backend/requirements.txt`
3. Install frontend packages from `frontend/package.json`
4. Create a `.env` file from `.env.example` if you do not already have one

Then open `.env` and add your key:

```env
GEMINI_API_KEY=your_key_here
HF_TOKEN=your_token_here
```

## Run

Use two terminals.

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

App: [http://localhost:5173/home](http://localhost:5173/home)  
Chat: [http://localhost:5173/chat](http://localhost:5173/chat)

In development, Vite proxies `/upload`, `/query`, `/methods`, and `/health` to the backend, so you do not need a separate frontend API URL.

## Project layout

```text
.
├── setup                 # one-command install
├── .env.example          # API key template
├── backend/
│   ├── main.py           # FastAPI entrypoint
│   ├── api/              # upload + query routes
│   ├── core/             # ingest, chunking, embed, retrieve, generate
│   └── requirements.txt
└── frontend/
    ├── src/              # React UI
    └── package.json
```

Uploaded PDFs, markdown, and the vector database live under `backend/storage/` and are ignored by git.

## Notes

- Only PDF uploads are supported.
- Answers depend on what you have ingested; an empty database returns a clear empty-state message.
- Keep `.env` private. Never commit real API keys.

## Author

[hasnat-nawaz](https://github.com/hasnat-nawaz) · [RAG-app](https://github.com/hasnat-nawaz/RAG-app)
