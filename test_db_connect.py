# test_db_connect.py
import os
from dotenv import load_dotenv
import sqlalchemy as sa

load_dotenv()
url = os.getenv("DATABASE_URL")
print("URL present:", bool(url))
if not url:
    raise SystemExit("DATABASE_URL not found. Check your .env file.")

# Create engine and try a simple query
engine = sa.create_engine(url, future=True)
with engine.connect() as conn:
    version = conn.exec_driver_sql("select version()").scalar_one()
    dbname = conn.exec_driver_sql("select current_database()").scalar_one()
    print("Connected OK")
    print("Database:", dbname)
    print("Server version:", version)
