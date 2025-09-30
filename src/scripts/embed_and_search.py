import os, uuid
from pathlib import Path
from typing import List

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

# config
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
MODEL_NAME = os.getenv("EMBED_MODEL")
EMBED_DIM = int(os.getenv("EMBED_DIM"))
TABLE = "demo_chunks"

if not DB_URL:
    raise SystemExit("DATABASE_URL missing in .env")

# Simple, beginner-friendly corpus across a few everyday topics
CORPUS: List[str] = [
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

QUERY = "Find sentences that are about animals."
TOP_K = int(os.getenv("TOP_K"))

def to_pgvector_literal(vec: np.ndarray) -> str:
    # pgvector accepts literal like: [0.12,-0.34,...]
    return "[" + ",".join(f"{float(x):.7f}" for x in vec) + "]"

def main():
    engine = create_engine(DB_URL, future=True, pool_pre_ping=True)

    # 1) embed sentences (normalized for cosine)
    model = SentenceTransformer(MODEL_NAME)
    corpus_vecs = model.encode(CORPUS, convert_to_numpy=True, normalize_embeddings=True)
    assert corpus_vecs.shape[1] == EMBED_DIM, f"got {corpus_vecs.shape[1]=}, expected {EMBED_DIM}"

    # 2) clear + insert
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {TABLE};"))
        ins = text(f"""
            INSERT INTO {TABLE} (id, text, embedding)
            VALUES (:id, :txt, CAST(:emb AS vector({EMBED_DIM})))
        """)
        for sent, vec in zip(CORPUS, corpus_vecs):
            conn.execute(ins, {"id": str(uuid.uuid4()), "txt": sent, "emb": to_pgvector_literal(vec)})
    print(f"✅ inserted {len(CORPUS)} rows")

    # 3) search
    qvec = model.encode([QUERY], convert_to_numpy=True, normalize_embeddings=True)[0]
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT text, 1 - (embedding <=> CAST(:q AS vector({EMBED_DIM}))) AS sim
            FROM {TABLE}
            ORDER BY embedding <=> CAST(:q AS vector({EMBED_DIM}))
            LIMIT :k
        """), {"q": to_pgvector_literal(qvec), "k": TOP_K}).mappings().all()

    print(f"\nQuery: {QUERY}")
    for r in rows:
        print(f"- sim={r['sim']:.3f} | {r['text']}")

if __name__ == "__main__":
    main()
