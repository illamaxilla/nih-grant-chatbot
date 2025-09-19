import os
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise SystemExit("DATABASE_URL missing in .env")

engine = sa.create_engine(DATABASE_URL, future=True)

def col_info(conn):
    row = conn.execute(text("""
        SELECT is_generated, generation_expression
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name='chunks' AND column_name='ts'
    """)).fetchone()
    if not row:
        return {"exists": False, "is_generated": False}
    is_gen = (row.is_generated or "").upper() == "ALWAYS" or (row.generation_expression is not None)
    return {"exists": True, "is_generated": is_gen}

with engine.begin() as conn:
    info = col_info(conn)

    # Case A: column doesn't exist → create plain tsvector + backfill + trigger + index
    if not info["exists"]:
        conn.execute(text("""
            ALTER TABLE chunks ADD COLUMN ts tsvector;
            UPDATE chunks SET ts = to_tsvector('english', coalesce(content,''));
            CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING GIN (ts);
            CREATE OR REPLACE FUNCTION chunks_tsvector_update() RETURNS trigger AS $$
            BEGIN
              NEW.ts := to_tsvector('english', coalesce(NEW.content,''));
              RETURN NEW;
            END
            $$ LANGUAGE plpgsql;
            DROP TRIGGER IF EXISTS chunks_tsvector_update_trg ON chunks;
            CREATE TRIGGER chunks_tsvector_update_trg
            BEFORE INSERT OR UPDATE OF content ON chunks
            FOR EACH ROW EXECUTE FUNCTION chunks_tsvector_update();
        """))
        mode = "created_plain"

    # Case B: column exists AND is generated → DO NOT touch the column; just ensure index
    elif info["is_generated"]:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING GIN (ts);
        """))
        # If an old trigger/function existed from a previous non-generated setup, it's harmless to leave;
        # but we won't create/update any triggers here.
        mode = "generated_detected"

    # Case C: column exists and is NOT generated → ensure backfill/trigger/index
    else:
        conn.execute(text("""
            UPDATE chunks
            SET ts = to_tsvector('english', coalesce(content,''))
            WHERE ts IS NULL OR ts = ''::tsvector;
            CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING GIN (ts);
            CREATE OR REPLACE FUNCTION chunks_tsvector_update() RETURNS trigger AS $$
            BEGIN
              NEW.ts := to_tsvector('english', coalesce(NEW.content,''));
              RETURN NEW;
            END
            $$ LANGUAGE plpgsql;
            DROP TRIGGER IF EXISTS chunks_tsvector_update_trg ON chunks;
            CREATE TRIGGER chunks_tsvector_update_trg
            BEFORE INSERT OR UPDATE OF content ON chunks
            FOR EACH ROW EXECUTE FUNCTION chunks_tsvector_update();
        """))
        mode = "existing_plain_synced"

    total = conn.execute(text("SELECT count(*) FROM chunks")).scalar_one()
    # "ts IS NOT NULL" is safe for both generated and non-generated
    with_ts = conn.execute(text("SELECT count(*) FROM chunks WHERE ts IS NOT NULL")).scalar_one()

print(f"✅ FTS ready ({mode}). chunks total={total}, with ts={with_ts}")
