import os
import sqlalchemy as sa
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise SystemExit("DATABASE_URL missing in .env")

engine = sa.create_engine(DATABASE_URL, future=True)

DDL = """
CREATE TABLE IF NOT EXISTS interactions (
  id              BIGSERIAL PRIMARY KEY,
  asked_at        TIMESTAMPTZ DEFAULT now(),
  question        TEXT,
  answer_preview  TEXT,               -- first ~500 chars of the answer
  foa_hint        TEXT,               -- e.g., RFA-DA-27-004 if detected
  sources         JSONB,              -- list of {title,url}
  latency_ms      INTEGER,
  ok              BOOLEAN,
  model           TEXT
);
CREATE INDEX IF NOT EXISTS idx_interactions_time ON interactions(asked_at DESC);
"""

with engine.begin() as conn:
    conn.execute(text(DDL))
    total = conn.execute(text("SELECT count(*) FROM interactions")).scalar_one()

print(f"✅ usage table ready. interactions={total}")
