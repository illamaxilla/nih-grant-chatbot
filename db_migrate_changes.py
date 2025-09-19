import os, sqlalchemy as sa
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv(".env")
engine = sa.create_engine(os.getenv("DATABASE_URL"), future=True)

DDL = """
-- minimal audit table for updates
CREATE TABLE IF NOT EXISTS changes (
  id              BIGSERIAL PRIMARY KEY,
  kind            TEXT NOT NULL,               -- 'document' | 'foa'
  url             TEXT,
  title           TEXT,
  change_summary  TEXT NOT NULL,
  seen_at         TIMESTAMPTZ DEFAULT now()
);

-- helpful indexes
CREATE INDEX IF NOT EXISTS idx_changes_seen_at ON changes(seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_changes_url ON changes((url) text_pattern_ops);
"""

with engine.begin() as conn:
    conn.execute(text(DDL))
print("✅ changes table ready.")
