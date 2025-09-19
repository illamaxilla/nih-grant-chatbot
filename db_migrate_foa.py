# db_migrate_foa.py
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
CREATE TABLE IF NOT EXISTS foas (
  id              BIGSERIAL PRIMARY KEY,
  foa_number      TEXT UNIQUE,         -- e.g., RFA-CA-25-001, PAR-OD-24-123, NOT-OD-24-999 (for notices)
  foa_type        TEXT,                -- RFA | PA | PAR | NOT
  title           TEXT,
  url             TEXT UNIQUE,
  activity_codes  TEXT[],              -- e.g., {R01,R21,U01}
  participating_ics TEXT[],            -- e.g., {NCI,NIAID}
  clinical_trial  TEXT,                -- Required | Optional | Not Allowed | None
  key_dates_json  JSONB,               -- {open_date: "...", due_dates: ["...","..."], expiration: "..."}
  last_seen       TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_foas_foa_number ON foas(foa_number);
CREATE INDEX IF NOT EXISTS idx_foas_url ON foas(url);
"""

with engine.begin() as conn:
    conn.execute(text(DDL))

print("✅ FOA metadata table ready.")
