from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel, Field
import json, subprocess, sys


app = FastAPI(title="StreamRAG Demo API")


class EmbedRequest(BaseModel):
    corpus: list[str] | None = None


class SearchRequest(BaseModel):
    query: str
    k: int = Field(5, ge=1, le=100)


def run_script(rel_path: str, payload: dict | None = None, expect_json: bool = True) -> dict | None:
    """Run a Python script by relative path from src/, send optional JSON stdin.

    If expect_json is True, parse and return JSON from stdout; otherwise return None on success.
    """
    base_dir = Path(__file__).resolve().parents[1]  # points to .../src
    script_path = base_dir / rel_path
    cmd = [sys.executable, str(script_path)]
    proc = subprocess.run(
        cmd,
        input=json.dumps(payload).encode("utf-8") if payload is not None else None,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        # Prefer stderr message if available
        msg = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(msg or f"Script failed: {rel_path} (code {proc.returncode})")
    if not expect_json:
        return None
    try:
        return json.loads(proc.stdout.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Invalid JSON from script {rel_path}: {e}")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/init")
def init_schema():
    # Reuse existing init_db.py; it prints text, not JSON
    run_script("scripts/init_db.py", expect_json=False)
    return {"ok": True}


@app.post("/embed")
def embed(req: EmbedRequest | None = None):
    payload = {"corpus": req.corpus} if (req is not None and req.corpus is not None) else {}
    out = run_script("scripts/embed.py", payload, expect_json=True)
    return out


@app.post("/search")
def search(req: SearchRequest):
    payload = {"query": req.query, "k": req.k}
    out = run_script("scripts/search.py", payload, expect_json=True)
    return out


@app.post("/answer")
def answer(req: SearchRequest):
    payload = {"query": req.query, "k": req.k}
    out = run_script("scripts/answer.py", payload, expect_json=True)
    return out

# Serve the simple HTML UI at root
static_dir = Path(__file__).parent / "web"
app.mount("/web", StaticFiles(directory=str(static_dir), html=True), name="web")

@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")
