import sys, os, json
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer


def to_pgvector_literal(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.7f}" for x in vec) + "]"


def main() -> int:
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL missing in .env", file=sys.stderr)
        return 2

    table = os.getenv("TABLE")
    embed_dim = int(os.getenv("EMBED_DIM"))
    model_name = os.getenv("EMBED_MODEL")
    default_query = os.getenv("QUERY")
    default_k = int(os.getenv("TOP_K"))

    stdin_data: Optional[str] = None
    try:
        if not sys.stdin.isatty():
            stdin_data = sys.stdin.read().strip() or None
    except Exception:
        pass

    if stdin_data:
        try:
            payload = json.loads(stdin_data)
            query = str(payload.get("query", default_query))
            k = int(payload.get("k", default_k))
        except Exception:
            query, k = default_query, default_k
    else:
        query, k = default_query, default_k

    engine = create_engine(db_url, future=True, pool_pre_ping=True)
    model = SentenceTransformer(model_name)
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
