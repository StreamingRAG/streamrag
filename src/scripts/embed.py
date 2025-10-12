import sys, os, uuid, json
from typing import List

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

    table = os.getenv("TABLE", "demo_chunks")
    embed_dim = int(os.getenv("EMBED_DIM", "384"))
    model_name = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Use a fixed default corpus (ignores any stdin for now)
    corpus: List[str] = [
        "The sky is blue on a clear day.",
        "Rain falls from clouds during a storm.",
        "Cats are small animals that like to sleep.",
        "Dogs are friendly pets that enjoy walks.",
        "Fish live in water and breathe through gills.",
        "Apples are sweet fruits that can be red or green.",
        "Bananas are yellow when they are ripe.",
        "Bread is made from flour, water, and yeast.",
        "Soccer is played with a round ball on a field.",
        "Basketball is played with a hoop and an orange ball.",
        "Cars use engines to move along roads.",
        "Trains run on tracks and can carry many people.",
    ]

    engine = create_engine(db_url, future=True, pool_pre_ping=True)
    model = SentenceTransformer(model_name)
    vecs = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
    if vecs.shape[1] != embed_dim:
        print(
            json.dumps({
                "ok": False,
                "error": f"Embedding dim mismatch: got {vecs.shape[1]}, expected {embed_dim}",
            }),
            file=sys.stderr,
        )
        return 3

    with engine.begin() as conn:
        # Ensure schema pieces exist
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
              id UUID PRIMARY KEY,
              text TEXT NOT NULL,
              created_at TIMESTAMPTZ DEFAULT now() NOT NULL,
              embedding vector({embed_dim})
            )
            """
        ))
        conn.execute(text(
            """
            CREATE INDEX IF NOT EXISTS demo_chunks_embedding_ivfflat
              ON demo_chunks USING ivfflat (embedding vector_cosine_ops)
              WITH (lists = 100)
            """
        ))
        # Demo simplicity: truncate then insert
        conn.execute(text(f"TRUNCATE {table};"))
        ins = text(f"""
            INSERT INTO {table} (id, text, embedding)
            VALUES (:id, :txt, CAST(:emb AS vector({embed_dim})))
        """)
        for sent, vec in zip(corpus, vecs):
            conn.execute(ins, {"id": str(uuid.uuid4()), "txt": sent, "emb": to_pgvector_literal(vec)})

    print(json.dumps({"ok": True, "inserted": len(corpus)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
