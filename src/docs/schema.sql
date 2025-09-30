CREATE EXTENSION IF NOT EXISTS vector;

-- Minimal table for demo content and embeddings
-- NOTE: embedding dimension is 384 to match the default MiniLM model
CREATE TABLE IF NOT EXISTS demo_chunks (
  id         UUID PRIMARY KEY,
  text       TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now() NOT NULL,
  embedding  vector(384)  -- must match EMBED_DIM
);

-- Create an IVFFlat index for faster approximate nearest neighbor search with cosine distance
-- Tune lists based on data size (e.g., sqrt(N) as a starting point). Adjust as needed.
CREATE INDEX IF NOT EXISTS demo_chunks_embedding_ivfflat
  ON demo_chunks USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- Optional: gather stats to help the planner
ANALYZE demo_chunks;
