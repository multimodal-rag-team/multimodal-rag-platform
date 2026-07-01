# Multimodal RAG Platform

A retrieval-augmented generation platform for technical documentation (research papers, product manuals, datasheets) that indexes and retrieves across **text, tables, and images** separately, with an agentic router that decides which modality a query needs before searching.

Built for documents where the answer often lives in a table or a figure, not just prose — standard text-only RAG discards that content or buries it under narrative chunks.

---

## Why this exists

Technical documents routinely combine prose, tables, and screenshots/diagrams. Text-only RAG pipelines either drop non-text content entirely or convert everything into one undifferentiated index, where tables and figures get drowned out by the much larger volume of narrative text. This platform keeps three separate indices and uses an LLM router to decide, per query, whether the answer should come from text, a table, an image, or a combination — then composes a single answer citing the source document and page.

## Architecture

```
User browser
     |
     v
Frontend (Next.js / React, port 3000)
     |
     v
FastAPI backend (port 8000)
     |
     v
Services layer (QueryService / IndexService / DocumentService)
     |
     +---------------------------+
     |  on upload                |  on query
     v                           v
PDF parser                  Router agent (classify route)
     |                           |
     v                           v
pgvector indexer            pgvector retriever (cosine search)
     |                           |
     +------------+--------------+
                  v
       PostgreSQL + pgvector
   (text_chunks / table_chunks / image_chunks)
                  |
        +---------+---------+
        v                   v
   OpenAI API           Langfuse
 (embeddings + LLM)    (tracing, optional)
```

A full architecture document with diagrams and step-by-step workflow explanations is included at [`docs/Multimodal_RAG_Architecture.pdf`](./docs/Multimodal_RAG_Architecture.pdf).

## How it works

**Indexing.** Uploaded PDFs are parsed page-by-page into typed blocks (heading, text, table, image). Table rows are converted into natural-language descriptions (LLM-summarised or deterministically formatted) while the original structured table is preserved in metadata. Each block is embedded with OpenAI's `text-embedding-3-small` and stored in one of three Postgres tables, each indexed with HNSW for cosine similarity search.

**Querying.** An LLM router classifies each incoming question into `text`, `table`, `image`, or `hybrid`, with a short reasoning string returned alongside the answer for transparency. The retriever searches the relevant table(s), assembles a labelled context (document id, page, similarity score), and an LLM composes the final answer with citations.

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16, React 19, Tailwind, shadcn/ui |
| Backend | FastAPI, Uvicorn |
| Parsing | unstructured, pdfplumber, PyMuPDF, Camelot |
| Embeddings / LLM | OpenAI (`text-embedding-3-small`, `gpt-4o-mini`) |
| Routing agent | LangChain structured output |
| Vector store | PostgreSQL + pgvector (HNSW, cosine distance) |
| Observability | Langfuse (optional) |
| Deployment | Docker Compose |

## Getting started

### Prerequisites

- Docker Desktop (with Compose v2)
- An OpenAI API key

### Run with Docker Compose (recommended)

```bash
git clone <repo-url>
cd multimodal-rag-platform
```

Add your OpenAI key and any other secrets to `MULTIMODAL_RAG/.env` (copy from `.env.example`):

```bash
cp MULTIMODAL_RAG/.env.example MULTIMODAL_RAG/.env
# edit MULTIMODAL_RAG/.env and set OPENAI_API_KEY
```

Then from the project root:

```bash
docker compose up --build
```

This builds and starts three containers — PostgreSQL with pgvector, the FastAPI backend, and the Next.js frontend — wired together on one network, with the database schema created automatically on first run.

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API docs | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

Subsequent runs (no code changes) only need:

```bash
docker compose up
```

### Run locally without Docker

**Backend**

```bash
cd MULTIMODAL_RAG
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

Ensure a local PostgreSQL instance has the `vector` extension available, set `DATABASE_URL` in `.env` accordingly, then:

```bash
python -m uvicorn api.main:app --reload
```

**Frontend**

```bash
cd multimodal-rag-ui
npm install
npm run dev
```

## Configuration

Key environment variables (see `MULTIMODAL_RAG/.env.example` for the full list):

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `OPENAI_API_KEY` | Required for embeddings, routing, and answer generation |
| `OPENAI_EMBEDDING_MODEL` | Default `text-embedding-3-small` |
| `OPENAI_ROUTER_MODEL` / `OPENAI_ANSWER_MODEL` / `OPENAI_TABLE_DESCRIPTION_MODEL` | Default `gpt-4o-mini` |
| `USE_LLM_TABLE_DESCRIPTIONS` | Toggle LLM table summarisation vs. deterministic formatting |
| `RAG_TOP_K` | Default number of chunks retrieved per query |
| `LANGFUSE_*` | Optional observability tracing |

## API overview

| Endpoint | Method | Description |
|---|---|---|
| `/index` | POST | Upload one or more PDFs for parsing and indexing |
| `/query` | POST | Ask a question; returns answer, route, reasoning, and sources |
| `/documents` | GET | List indexed documents with chunk counts per modality |
| `/health` | GET | Liveness check |

Full interactive documentation is available at `/docs` once the backend is running.

## Evaluation

A golden-dataset evaluation harness lives in `MULTIMODAL_RAG/evaluation/`. It runs a curated set of question/answer pairs — spanning text-, table-, and image-routed queries across 50 source documents — against the live `/query` endpoint, scoring each response with token-overlap F1, route-classification accuracy, and an LLM-as-judge score (0–5), and produces a comparative Markdown and JSON report.

```bash
cd MULTIMODAL_RAG
python evaluation/evaluate.py --llm-judge
```

See `evaluation/reports/` for the latest run.

## Project structure

```
multimodal-rag-platform/
├── docker-compose.yml
├── docs/
│   └── Multimodal_RAG_Architecture.pdf
├── MULTIMODAL_RAG/              # backend
│   ├── api/                     # FastAPI app, routes, schemas, services
│   ├── agent/                   # router agent (query classification)
│   ├── parser/                  # PDF parsing (text/table/image extraction)
│   ├── indexer/                 # embedding + pgvector storage, schema.sql
│   ├── retriever/                # pgvector similarity search
│   ├── config/                  # settings
│   ├── evaluation/               # golden dataset + evaluation harness
│   └── Dockerfile
└── multimodal-rag-ui/            # frontend
    ├── app/
    ├── components/
    ├── hooks/
    ├── lib/
    └── Dockerfile
```

## Design notes

The decision to keep three separate indices instead of one merged index, and to use an LLM router rather than keyword heuristics, is covered in detail — along with trade-offs around hybrid-route latency — in the [architecture document](./docs/Multimodal_RAG_Architecture.pdf).
