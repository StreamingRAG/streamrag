import sys, os, json
from typing import Optional
from sqlalchemy import text
from ._util import to_pgvector_literal, get_sentence_model, init_engine_table_dim


def main() -> int:
    # Engine/table/dim from util; centralized error handling within init
    engine, table, embed_dim = init_engine_table_dim(exit_on_error=True)
    model = get_sentence_model()
    default_query = os.getenv("QUERY")
    default_k = int(os.getenv("TOP_K"))

    # Read optional stdin JSON once
    try:
        stdin_data: Optional[str] = None if sys.stdin.isatty() else (sys.stdin.read().strip() or None)
    except (AttributeError, OSError):
        stdin_data = None

    payload: dict = {}
    if stdin_data:
        try:
            payload = json.loads(stdin_data) or {}
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            payload = {}

    raw_q = payload.get("query", default_query)
    query = (str(raw_q).strip() if raw_q is not None else "")
    k = int(payload.get("k", default_k)) if isinstance(payload, dict) else default_k

    if not query:
        print(json.dumps({
            "ok": False,
            "error": "No query provided. Please include 'query' in the request.",
        }))
        return 0

    qvec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT text, 1 - (embedding <=> CAST(:q AS vector({embed_dim}))) AS sim
            FROM {table}
            ORDER BY embedding <=> CAST(:q AS vector({embed_dim}))
            LIMIT :k
        """), {"q": to_pgvector_literal(qvec), "k": k}).mappings().all()

    print(json.dumps({
        "ok": True,
        "query": query,
        "k": k,
        "results": [{"text": r["text"], "similarity": float(r["sim"]) } for r in rows],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
