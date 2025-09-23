# main.py
from fastapi import FastAPI, HTTPException, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
import sqlalchemy as sa
import sqlalchemy.exc as sa_exc
from sqlalchemy import text
from pgvector.psycopg import register_vector  # psycopg v3 adapter
import os, time, random, re, json, math, datetime
from typing import List, Dict, Any, Optional, Tuple

# ---------------- Env ----------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
RAW_DATABASE_URL = os.getenv("DATABASE_URL") or ""
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

# Embeddings config (server can override without code changes)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small").strip()
EMBED_FALLBACKS = [m.strip() for m in os.getenv(
    "EMBED_FALLBACKS",
    "text-embedding-3-large"
).split(",") if m.strip()]

# NIH-only retrieval domain whitelist (comma-separated suffixes)
DOMAINS_WHITELIST = [d.strip().lower() for d in os.getenv(
    "DOMAINS_WHITELIST", "nih.gov,cancer.gov"
).split(",") if d.strip()]

def _normalize_db_url(url: str) -> str:
    # Convert older "postgres://" to "postgresql://"
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    # Ensure SQLAlchemy uses psycopg v3 driver (NOT psycopg2)
    if url.startswith("postgresql://") and "+psycopg" not in url and "+psycopg2" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url

DATABASE_URL = _normalize_db_url(RAW_DATABASE_URL)

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL missing")

# ---------------- Clients / Engine ----------------
client = OpenAI(api_key=OPENAI_API_KEY)

# Harden the pool: pre-ping, recycle, lifo, etc. (helps Render & Supabase poolers)
engine = sa.create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,         # validate connections before use
    pool_recycle=1800,          # recycle every 30 min
    pool_size=5,                # modest base pool
    max_overflow=10,            # allow bursts
    pool_timeout=30,            # wait up to 30s for a connection
    pool_use_lifo=True,         # reduce stampedes
    connect_args={
        "prepare_threshold": 0,     # avoid prepared statement churn
        "channel_binding": "disable"
    },
)

# ---------------- Admin auth ----------------
def require_admin(x_api_key: str = Header(None)):
    """
    If ADMIN_API_KEY is set, require header:
      x-api-key: <ADMIN_API_KEY>
    """
    if not ADMIN_API_KEY:
        return  # auth disabled if not set
    if x_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---------------- App & static ----------------
app = FastAPI(title="NIH Grants Chatbot (MVP)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Serve the web UI at /app (files in ./static)
app.mount("/app", StaticFiles(directory="static", html=True), name="static")

# Also serve root "/" for convenience
@app.get("/", response_class=HTMLResponse)
def root_index():
    return FileResponse("static/index.html")

# ---------------- Health ----------------
@app.get("/health")
def health():
    ok = True
    db_ok = True
    msg = "ok"
    try:
        with engine.begin() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        ok = False
        db_ok = False
        msg = f"db_error: {e.__class__.__name__}"
    return {"ok": ok, "db_ok": db_ok, "message": msg}

# ---------------- Models (simple check) ----------------
@app.get("/models")
def list_models():
    try:
        data = client.models.list()
        return {"count": len(data.data), "models": [m.id for m in data.data[:10]]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"OpenAI error: {type(e).__name__}")

# ---------------- Simple /ask (no RAG) ----------------
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

SYSTEM_PROMPT_GENERIC = (
    "You are a concise assistant. If unsure, say so briefly. "
    "Keep answers short. No markdown code fences."
)

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    for attempt in range(5):
        try:
            chat = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_GENERIC},
                    {"role": "user", "content": req.question.strip()[:4000]},
                ],
                temperature=0.2,
            )
            return AskResponse(answer=chat.choices[0].message.content.strip())
        except Exception as e:
            msg = str(e)
            if attempt < 4 and any(x in msg for x in ["RateLimit", "429", "ServiceUnavailable", "Timeout"]):
                time.sleep((0.5 * (2 ** attempt)) + random.uniform(0, 0.25))
                continue
            raise HTTPException(status_code=503, detail=f"LLM error: {type(e).__name__}")

# ---------------- FOA endpoints ----------------
FOA_RE = re.compile(r"\b(RFA|PA|PAR)-[A-Z]{2,}-\d{2}-\d{3}[A-Z]?\b", re.IGNORECASE)
NOTICE_RE = re.compile(r"\bNOT-[A-Z]{2,}-\d{2}-\d{3}\b", re.IGNORECASE)

@app.get("/foa/search")
def search_foa(q: str):
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT foa_number, foa_type, title, url
            FROM foas
            WHERE foa_number ILIKE :q OR title ILIKE :q
            ORDER BY last_seen DESC
            LIMIT 25
        """), {"q": f"%{q}%"}).fetchall()
    return [{"foa_number": r.foa_number, "foa_type": r.foa_type, "title": r.title, "url": r.url} for r in rows]

@app.get("/foa/{foa_number}")
def get_foa(foa_number: str):
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT foa_number, foa_type, title, url, activity_codes, participating_ics, clinical_trial, key_dates_json
            FROM foas WHERE upper(foa_number)=upper(:n)
        """), {"n": foa_number}).fetchone()
    if not row:
        raise HTTPException(404, f"FOA '{foa_number}' not found (ingest may not have seen it yet).")
    return {
        "foa_number": row.foa_number,
        "foa_type": row.foa_type,
        "title": row.title,
        "url": row.url,
        "activity_codes": row.activity_codes or [],
        "participating_ics": row.participating_ics or [],
        "clinical_trial": row.clinical_trial,
        "key_dates": (json.loads(row.key_dates_json) if row.key_dates_json else {}),
    }

def get_foa_meta(foa_number: str) -> Optional[Dict[str, Any]]:
    with engine.begin() as conn:
        r = conn.execute(text("""
            SELECT foa_number, foa_type, title, url, activity_codes, participating_ics, clinical_trial, key_dates_json
            FROM foas WHERE upper(foa_number)=upper(:n)
        """), {"n": foa_number}).fetchone()
    if not r:
        return None
    kd = json.loads(r.key_dates_json) if r.key_dates_json else {}
    return {
        "foa_number": r.foa_number,
        "title": r.title,
        "url": r.url,
        "activity_codes": r.activity_codes or [],
        "participating_ics": r.participating_ics or [],
        "clinical_trial": r.clinical_trial,
        "key_dates": kd,
    }

# ---------------- Embeddings helpers ----------------
def _embed_once(model_name: str, text_in: str) -> Tuple[List[float], str]:
    resp = client.embeddings.create(model=model_name, input=[text_in])
    return resp.data[0].embedding, model_name

def embed_query(q: str) -> Tuple[List[float], str]:
    """
    Try primary model, then fallbacks. Return (vector, model_used).
    Raise on final failure; caller may catch and continue without embeddings.
    """
    models_to_try = [EMBED_MODEL] + [m for m in EMBED_FALLBACKS if m != EMBED_MODEL]
    last_err = None
    for m in models_to_try:
        try:
            vec, used = _embed_once(m, q)
            return vec, used
        except Exception as e:
            last_err = e
            print(f"[embed_query] failed model={m}: {e.__class__.__name__}: {e}")
            continue
    raise last_err if last_err else RuntimeError("embedding_failed")

def _to_list(vec):
    try:
        return list(vec)
    except Exception:
        try:
            return vec.tolist()
        except Exception:
            return [float(x) for x in vec]

def _cosine(a, b):
    num = sum(x*y for x, y in zip(a, b))
    da = math.sqrt(sum(x*x for x in a)) + 1e-12
    db = math.sqrt(sum(y*y for y in b)) + 1e-12
    return num / (da * db)

def mmr_rerank(q_vec, cand_vecs, k=6, lambda_mult=0.7):
    if not cand_vecs: return []
    selected, remaining = [], list(range(len(cand_vecs)))
    rel = [ _cosine(q_vec,
