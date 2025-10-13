import sys, uuid, json
from typing import List
from sqlalchemy import text
from ._util import to_pgvector_literal, get_sentence_model, init_engine_table_dim


def main() -> int:
    # Engine/table/dim from util; centralized error handling within init
    engine, table, embed_dim = init_engine_table_dim(exit_on_error=True)
    # Model from util; EMBED_MODEL must be set in environment
    model = get_sentence_model()

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
        # Rely on /init to create extension/table/index. Only mutate data here.
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
