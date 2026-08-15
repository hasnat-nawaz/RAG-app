# Frontend instructions

Vite + React UI for the local RAG FastAPI backend.

## Prerequisites

- Node.js 18+ (npm or pnpm)
- Backend running at `http://127.0.0.1:8000` (see backend setup)

## Install

```bash
cd frontend
npm install
```

## Run (development)

```bash
cd frontend
npm run dev
```

Open **http://localhost:5173/home** (root `/` redirects there). Chat lives at **http://localhost:5173/chat**.

In dev, Vite proxies `/upload` and `/query` to the backend, so you usually do **not** need a `.env` file.

## Optional: point at a custom API URL

Create `frontend/.env`:

```env
VITE_API_URL=http://127.0.0.1:8000
```

Then restart `npm run dev`. The backend must allow CORS for the Vite origin (`http://localhost:5173`).

## Production build

```bash
cd frontend
npm run build
npm run preview
```

## What the UI talks to

| UI action | Backend |
|-----------|---------|
| Upload PDF | `POST /upload` (`multipart/form-data`, field `file`) |
| Ask question | `POST /query` (`{ query, hybrid, hyde, top_k }`) |

See root `API.md` for full request/response shapes.

## Layout (matches architecture)

```
frontend/
├── package.json
├── vite.config.js
├── index.html
├── instructions.md
└── src/
    ├── App.jsx
    ├── main.jsx
    ├── styles.css
    ├── api/
    │   └── client.js
    └── components/
        ├── UploadPanel.jsx
        ├── QueryPanel.jsx
        ├── MethodSelector.jsx
        ├── ResultsPanel.jsx
        ├── MetricsPanel.jsx
        └── RetrievedChunksPanel.jsx
```

## Notes

- PDF uploads only.
- At least one of Hybrid / HyDE must stay selected.
- Upload and query can take a long time; the UI uses long timeouts.
- `RetrievedChunksPanel` is a placeholder — `/query` does not return chunk text yet.
