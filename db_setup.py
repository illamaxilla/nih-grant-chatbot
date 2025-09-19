# db_setup.py
import os
import sqlalchemy as sa
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL missing from .env")

engine = sa.create_engine(DATABASE_URL, future=True)

DDL = """
-- Make sure the pgvector extension exists
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per source URL
CREATE TABLE IF NOT EXISTS documents (
  id           BIGSERIAL PRIMARY KEY,
  url          TEXT UNIQUE NOT NULL,
  title        TEXT,
  fetched_at   TIMESTAMPTZ,
  checksum     TEXT
);

-- Chunked text from each document
CREATE TABLE IF NOT EXISTS chunks (
  id           BIGSERIAL PRIMARY KEY,
  doc_id       BIGINT REFERENCES documents(id) ON DELETE CASCADE,
  ordinal      INT NOT NULL,
  heading_path TEXT,
  content      TEXT NOT NULL,
  token_count  INT DEFAULT 0
);

-- Vector embeddings per chunk (text-embedding-3-small → 1536 dims)
CREATE TABLE IF NOT EXISTS embeddings (
  chunk_id     BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  embedding    vector(1536)
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_chunks_docid ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_docs_url ON documents(url);

-- Vector index (IVFFlat with cosine distance) - 1536 dims is OK for IVFFlat
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes WHERE indexname = 'idx_embeddings_vec'
  ) THEN
    EXECUTE 'CREATE INDEX idx_embeddings_vec ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);';
  END IF;
END$$;
"""

with engine.begin() as conn:
    conn.execute(text(DDL))

print("✅ Database schema created (or already existed).")
