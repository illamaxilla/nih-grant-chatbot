# check_foas.py
import os
import sqlalchemy as sa
from sqlalchemy import text
from dotenv import load_dotenv

# Explicitly load .env from the current directory
load_dotenv(dotenv_path=".env")

db_url = os.getenv("DATABASE_URL")
if not db_url:
    raise SystemExit("DATABASE_URL not found. Is .env in this folder?")

engine = sa.create_engine(db_url, future=True)
with engine.begin() as c:
    n = c.execute(text("select count(*) from foas")).scalar_one()
    print("FOA rows:", n)
    rows = c.execute(text("""
        select foa_number, title, url, last_seen
        from foas
        order by last_seen desc
        limit 10
    """)).fetchall()
    for r in rows:
        print(f"- {r.foa_number or '(none)'} | { (r.title or '')[:90] } | {r.url}")
