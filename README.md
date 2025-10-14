# streamrag <img alt="CI" src="https://github.com/StreamingRAG/streamrag/actions/workflows/ci.yml/badge.svg">

Minimal RAG demo on PostgreSQL + pgvector, FastAPI, and local LLM via Ollama.

## Quick start (Windows PowerShell)

1) Install and start Ollama

```powershell
winget install Ollama.Ollama -s winget
ollama pull gemma3
```

2) Configure environment

Copy `.env.example` to `.env` and set values (DB URL, model, templates).

3) Install deps and run API

Simplest (Python):

```powershell
uv run python run.py
```

Alternative (original uvicorn CLI):

```powershell
uv sync
uv run uvicorn --app-dir src streamrag.api:app --reload
```

4) Open http://127.0.0.1:8000 and click:

Notes:
- Answer JSON includes `mode` and `max_similarity` to show grounded vs general.
- Templates live in `src/docs/prompt_grounded.txt` and `src/docs/prompt_general.txt`.