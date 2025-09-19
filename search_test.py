# search_test.py
import os
import sqlalchemy as sa
from sqlalchemy import text
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector, Vector

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
engine = sa.create_engine(DATABASE_URL, future=True)
client = OpenAI(api_key=OPENAI_API_KEY)

def search(query: str, k: int = 5):
    qe = client.embeddings.create(
        model="text-embedding-3-small",  # 1536-dim to match schema
        input=[query]
    ).data[0].embedding

    with engine.begin() as conn:
        raw = conn.connection.driver_connection
        register_vector(raw)
        rows = conn.execute(text("""
            SELECT c.content, d.title, d.url
            FROM embeddings e
            JOIN chunks c ON c.id = e.chunk_id
            JOIN documents d ON d.id = c.doc_id
            ORDER BY e.embedding <-> :qe
            LIMIT :k
        """), {"qe": Vector(qe), "k": k}).fetchall()

    return rows

if __name__ == "__main__":
    for row in search("What are the NIH page limits for an R01?"):
        print("\n---")
        print(row.title)
        print(row.url)
        print(row.content[:400], "…")
