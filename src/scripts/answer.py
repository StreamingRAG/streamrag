import os, sys, json, subprocess
from typing import List, Dict, Any

from dotenv import load_dotenv

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
    except Exception:
        pass
    return None


def run_search(query: str, k: int) -> Dict[str, Any]:
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "search.py")]
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
    Falls back to a built-in prompt if the selected file is missing.
    """
    # Choose mode-specific template only
    template = _load_template(
        os.getenv("PROMPT_TEMPLATE_GENERAL_PATH") if mode == "general" else os.getenv("PROMPT_TEMPLATE_GROUNDED_PATH")
    )
    context_block = _format_context(results)
    if template:
        return (
            template
            .replace("{{MODE}}", mode)
            .replace("{{CONTEXT}}", context_block)
            .replace("{{QUESTION}}", query)
        )

    if mode == "general":
        parts = [
            f"Mode: {mode}",
            context_block,
            "",
            f"Question: {query}",
            "",
            "Instructions:",
            "- You may use your general knowledge to answer.",
            "- Prefer any relevant information from the context above, citing snippet numbers like [2], [5].",
            "- If there isn't enough information for a confident answer, please give your best guess",
            "- Be clear and concise.",
        ]
        return "\n".join(parts)

    # grounded (default) tuned for concise analysis and anomaly (why-not) explanations
    parts = [
        f"Mode: {mode}",
        context_block,
        "",
        f"Question: {query}",
        "",
        "Instructions:",
        "- Base your answer strictly on the context snippets above.",
        "- Identify which snippets match and explain why (traits, classification, behavior).",
        "- Briefly note any top snippets that do NOT match and why (anomalies/out-of-scope).",
        "- Use citations with snippet numbers like [2], [5].",
        "- Be concise: 3-6 bullets total, each 1-2 short sentences.",
        "- If there isn't enough information, reply: I don't know.",
        "",
        "Output format: Bulleted list with citations. Optionally end with a one-line summary.",
    ]
    return "\n".join(parts)


def call_ollama(prompt: str) -> str:
    try:
        import ollama  # type: ignore
    except Exception as e:
        raise RuntimeError("Ollama client not installed. Please run: uv add ollama") from e

    # Read environment variables directly and cast
    model = os.getenv("OLLAMA_MODEL")
    temperature = float(os.getenv("RESPONSE_TEMPERATURE"))
    num_ctx = int(os.getenv("NUM_CTX"))
    num_predict = int(os.getenv("NUM_PREDICT"))
    try:
        # Use chat API with a guiding system message; allow general knowledge if mode permits (conveyed in prompt)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful assistant. In grounded mode, answer using only the provided context snippets. "
                    "In general mode, you may answer using your general knowledge even if the context is irrelevant, "
                    "while preferring and citing any relevant snippets when available. If you truly cannot answer, say 'I don't know'."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        resp = ollama.chat(
            model=model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        )
        # Prefer attribute access (ChatResponse), fallback to dict
        text = getattr(getattr(resp, "message", {}), "content", None)
        if not text and isinstance(resp, dict):
            text = resp.get("message", {}).get("content")
        return text or ""
    except Exception as e:
        raise RuntimeError(
            "Ollama API error. Ensure Ollama is running and the model is pulled (e.g., 'ollama pull gemma3')."
        ) from e


def main() -> int:
    load_dotenv()

    # Read stdin JSON: {"query": str, "k": int}
    data = sys.stdin.read().strip()
    try:
        payload = json.loads(data) if data else {}
    except Exception:
        payload = {}
    query = str(payload.get("query", "Find sentences that are about animals.")).strip()
    k = int(payload.get("k", 5))

    search_out = run_search(query, k)
    results = search_out.get("results", []) if isinstance(search_out, dict) else []
    # Choose mode based on context strength: default GENERAL unless threshold for grounded is met
    max_sim = max((float(r.get("similarity", 0.0)) for r in results), default=0.0)
    threshold = float(os.getenv("CONTEXT_THRESHOLD"))
    mode = "general"
    if max_sim >= threshold:
        mode = "grounded"

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
