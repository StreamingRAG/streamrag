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


def run_script(module: str, payload: dict | None = None, expect_json: bool = True) -> dict | None:
    """Run a Python module (e.g., 'scripts.search') and optionally parse JSON output.

    If expect_json is True, parse and return JSON from stdout; otherwise return None on success.
    """
    cmd = [sys.executable, "-m", module]
    # Ensure src-layout packages (e.g., 'scripts') are importable by running from the src directory
    src_dir = Path(__file__).resolve().parent.parent  # .../src
    proc = subprocess.run(
        cmd,
        input=json.dumps(payload).encode("utf-8") if payload is not None else None,
        capture_output=True,
        check=False,
        cwd=str(src_dir),
    )
    if proc.returncode != 0:
        # Prefer stderr message if available
        msg = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(msg or f"Script failed: {module} (code {proc.returncode})")
    if not expect_json:
        return None
    try:
        return json.loads(proc.stdout.decode("utf-8"))
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        raise RuntimeError(f"Invalid JSON from script {module}: {e}") from e


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/init")
def init_schema() -> dict:
    # Reuse existing init_db; it prints text, not JSON
    run_script("scripts.init_db", expect_json=False)
    return {"ok": True}


@app.post("/embed")
def embed(req: EmbedRequest | None = None) -> dict:
    payload = {"corpus": req.corpus} if (req is not None and req.corpus is not None) else {}
    out = run_script("scripts.embed", payload, expect_json=True)
    return out


@app.post("/search")
def search(req: SearchRequest) -> dict:
    payload = {"query": req.query, "k": req.k}
    out = run_script("scripts.search", payload, expect_json=True)
    return out


@app.post("/answer")
def answer(req: SearchRequest) -> dict:
    payload = {"query": req.query, "k": req.k}
    out = run_script("scripts.answer", payload, expect_json=True)
    return out

# Serve the simple HTML UI at root
static_dir = Path(__file__).parent / "web"
app.mount("/web", StaticFiles(directory=str(static_dir), html=True), name="web")

@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")
