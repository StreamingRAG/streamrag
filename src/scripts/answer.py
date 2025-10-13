import json
import os
import subprocess
import sys

import ollama
from pathlib import Path
from typing import List, Dict, Any
from ._util import get_ollama_config

def _format_context(results: List[Dict[str, Any]]) -> str:
    lines: List[str] = ["Context snippets:"]
    for i, r in enumerate(results, 1):
        txt = str(r.get("text", "")).strip()
        lines.append(f"[{i}] {txt}")
    return "\n".join(lines)


def _load_template(path: str | None) -> str | None:
    if not path:
        return None
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except OSError:
        # Ignore file-related errors and fall back to default prompt
        pass
    return None


def run_search(query: str, k: int) -> Dict[str, Any]:
    # Prefer running as a module to keep relative imports working
    cmd = [sys.executable, "-m", "scripts.search"]
    payload = json.dumps({"query": query, "k": k}).encode("utf-8")
    proc = subprocess.run(cmd, input=payload, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore") or "search failed")
    return json.loads(proc.stdout.decode("utf-8"))


def build_prompt(query: str, results: List[Dict[str, Any]], *, mode: str = "grounded") -> str:
    """
    Build a user prompt using exactly two templates:
      - PROMPT_TEMPLATE_GROUNDED_PATH
      - PROMPT_TEMPLATE_GENERAL_PATH
    Placeholders: {{MODE}}, {{CONTEXT}}, {{QUESTION}}
    Loads templates from src/docs and does not use in-code fallbacks.
    """
    # Resolve template path under src/docs in a single expression
    tpl_path = Path(__file__).resolve().parents[1] / "docs" / (
        "prompt_general.txt" if mode == "general" else "prompt_grounded.txt"
    )
    template = _load_template(str(tpl_path))
    if not template:
        raise RuntimeError(f"Missing prompt template: {tpl_path}")
    return (
        template
        .replace("{{MODE}}", mode)
        .replace("{{CONTEXT}}", _format_context(results))
        .replace("{{QUESTION}}", query)
    )


def call_ollama(prompt: str) -> str:
    # Read Ollama configuration via shared helper (validates and casts)
    cfg = get_ollama_config()
    # Use chat API with a guiding system message; allow general knowledge if mode permits (conveyed in prompt)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful assistant. In grounded mode, answer using only the provided context snippets. "
                "In general mode, you may answer using your general knowledge even if the context is irrelevant, "
                "while preferring and citing any relevant snippets when applicable. Answer with your best guess if unknown."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    try:
        resp = ollama.chat(
            model=cfg["model"],
            messages=messages,
            options={
                "temperature": cfg["temperature"],
                "num_ctx": cfg["num_ctx"],
                "num_predict": cfg["num_predict"],
            },
        )
    except Exception as e:  # noqa: BLE001 - third-party may raise varied exceptions
        raise RuntimeError(
            "Ollama API error. Ensure Ollama is running and the model is pulled (e.g., 'ollama pull gemma3')."
        ) from e

    # Prefer attribute access (ChatResponse), fallback to dict
    text = getattr(getattr(resp, "message", {}), "content", None)
    if not text and isinstance(resp, dict):
        # Be specific on parsing-related issues to avoid very broad catches
        try:
            text = resp.get("message", {}).get("content")  # type: ignore[assignment]
        except (AttributeError, TypeError, ValueError):
            text = None
    return text or ""


def main() -> int:

    # Read stdin JSON: {"query": str, "k": int}
    data = sys.stdin.read().strip()
    try:
        payload = json.loads(data) if data else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        payload = {}
    raw_q = payload.get("query")
    query = (str(raw_q).strip() if raw_q is not None else "")
    if not query:
        print(json.dumps({
            "ok": False,
            "error": "No query provided. Please include a 'query' in the request.",
        }))
        return 0
    k = int(payload.get("k", 5))

    search_out = run_search(query, k)
    results = search_out.get("results", []) if isinstance(search_out, dict) else []
    # Choose mode based on context strength: default GENERAL unless threshold for grounded is met
    max_sim = max((float(r.get("similarity", 0.0)) for r in results), default=0.0)
    threshold = float(os.getenv("CONTEXT_THRESHOLD"))
    mode = "grounded" if max_sim >= threshold else "general"

    prompt = build_prompt(query, results, mode=mode)
    answer = call_ollama(prompt)

    out = {
        "ok": True,
        "answer": answer,
        "mode": mode,
        "max_similarity": max_sim,
        "sources": [{"id": i+1, "text": r.get("text", ""), "similarity": r.get("similarity")}
                    for i, r in enumerate(results)],
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
