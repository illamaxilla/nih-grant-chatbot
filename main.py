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
from typing import List, Dict, Any, Optional

# ---------------- Env ----------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
RAW_DATABASE_URL = os.getenv("DATABASE_URL") or ""
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

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

# Supabase pooler is picky; these connect args avoid prepare/channel-binding issues.
engine = sa.create_engine(
    DATABASE_URL,
    future=True,
    connect_args={"prepare_threshold": 0, "channel_binding": "disable"},
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

SYSTEM_PROMPT_GENERIC = "You are a concise assistant. If unsure, say so briefly. Keep answers short. No markdown backticks."

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

# ---------------- RAG helpers (Hybrid + MMR) ----------------
def embed_query(q: str) -> List[float]:
    return client.embeddings.create(model="text-embedding-3-small", input=[q]).data[0].embedding

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
    rel = [ _cosine(q_vec, v) for v in cand_vecs ]
    while remaining and len(selected) < k:
        if not selected:
            i = max(remaining, key=lambda idx: rel[idx])
            selected.append(i); remaining.remove(i); continue
        def score(idx):
            max_sim_to_sel = max(_cosine(cand_vecs[idx], cand_vecs[j]) for j in selected)
            return lambda_mult * rel[idx] - (1 - lambda_mult) * max_sim_to_sel
        i = max(remaining, key=score)
        selected.append(i); remaining.remove(i)
    return selected

def _candidate_rows(
    conn,
    qe: List[float],
    topN: int,
    where_extra: str = "",
    params: dict | None = None,
    policy_bias: bool = False,
):
    """
    Always CAST the param to vector to avoid 'vector <-> double precision[]' errors.
    Optionally bias toward policy/how-to/standard-due-dates pages for generic queries.
    """
    params = params or {}

    base_expr = "e.embedding <-> CAST(:qe AS vector)"
    if policy_bias:
        # Lower is better; subtract a small epsilon for desired pages.
        bias_case = """
            + CASE
                WHEN d.url ILIKE '%/how-to-apply-application-guide/%' THEN -0.08
                WHEN d.url ILIKE '%/grants-process/%'                 THEN -0.06
                WHEN d.url ILIKE '%/due-dates%'                       THEN -0.12
                WHEN d.url ILIKE '%/nihgps/%'                         THEN -0.05
                WHEN d.title ILIKE '%standard due date%'              THEN -0.12
                WHEN d.title ILIKE '%application guide%'              THEN -0.08
                ELSE 0.0
              END
        """
        order_expr = f"{base_expr} {bias_case}"
    else:
        order_expr = base_expr

    sql = f"""
        SELECT c.content, d.title, d.url, e.embedding AS emb
        FROM embeddings e
        JOIN chunks c ON c.id = e.chunk_id
        JOIN documents d ON d.id = c.doc_id
        {where_extra}
        ORDER BY {order_expr}
        LIMIT :n
    """
    rows = conn.execute(text(sql), {"qe": qe, "n": topN, **params}).fetchall()
    return [{"content": r.content, "title": r.title, "url": r.url, "emb": _to_list(r.emb)} for r in rows]

def _keyword_fallback_rows(conn, query: str, k: int = 6) -> List[Dict[str, Any]]:
    """
    LAST-RESORT: If vector/FTS yields nothing, fall back to simple keyword filters
    over high-signal NIH policy pages. No embeddings required.
    """
    # Build a few loose patterns; include due-dates, application guide, nihgps, peer review.
    patterns = [
        "%/due-dates-and-submission-policies/%",
        "%/due-dates.htm%",
        "%/how-to-apply-application-guide/%",
        "%/nihgps/%",
        "%/peer-review.htm%",
    ]
    # Also search for the user's key words in title/content.
    words = [w for w in re.findall(r"[A-Za-z0-9]{3,}", query)][:5]
    like_params = {f"p{i}": p for i, p in enumerate(patterns)}
    wc = []
    for i in range(len(patterns)):
        wc.append(f"d.url ILIKE :p{i}")
    # Add up to 5 keywords into title/content
    for j, w in enumerate(words):
        like_params[f"w{j}"] = f"%{w}%"
        wc.append(f"d.title ILIKE :w{j}")
        wc.append(f"c.content ILIKE :w{j}")

    where_clause = "WHERE " + " OR ".join(wc) if wc else ""
    sql = f"""
        SELECT c.content, d.title, d.url
        FROM chunks c
        JOIN documents d ON d.id = c.doc_id
        {where_clause}
        ORDER BY
            CASE
              WHEN d.url ILIKE '%/due-dates%' THEN 0
              WHEN d.url ILIKE '%/nihgps/%' THEN 1
              WHEN d.url ILIKE '%/how-to-apply-application-guide/%' THEN 2
              ELSE 3
            END,
            length(c.content) DESC
        LIMIT :n
    """
    rows = conn.execute(text(sql), {**like_params, "n": max(12, k * 4)}).fetchall()
    out = [{"content": r.content, "title": r.title, "url": r.url, "emb": None} for r in rows]
    # Return only k items (no MMR here since we may not have embeddings)
    return out[:k]

def retrieve_top_chunks(query: str, k: int = 6, foa_hint: str = None):
    """
    1) Try FOA-constrained search (if hint present)
    2) Else hybrid: FTS pre-filter + vector rank
    3) Else pure vector
    4) Else keyword-only fallback over policy pages
    Then MMR to diversify results (when embeddings exist).
    """
    qe: Optional[List[float]] = None
    try:
        qe = embed_query(query)  # Python list
    except Exception:
        qe = None  # allow fallback path

    topN = max(12, k * 4)

    try:
        with engine.begin() as conn:
            raw = conn.connection.driver_connection
            register_vector(raw)

            # 1) FOA-focused (no policy bias)
            if qe is not None and foa_hint:
                like = f"%{foa_hint.upper()}%"
                cands = _candidate_rows(
                    conn, qe, topN,
                    "WHERE d.url ILIKE :like OR d.title ILIKE :like",
                    {"like": like},
                    policy_bias=False,
                )
                if cands:
                    vecs = [c["emb"] for c in cands]
                    keep = mmr_rerank(qe, vecs, k=k, lambda_mult=0.7)
                    return [cands[i] for i in keep]

            # 2) FTS + vector (bias ON for generic queries)
            cands: List[Dict[str, Any]] = []
            if qe is not None:
                try:
                    cands = _candidate_rows(
                        conn, qe, topN,
                        "WHERE c.ts @@ plainto_tsquery('english', :ftsq)",
                        {"ftsq": query},
                        policy_bias=(foa_hint is None),
                    )
                except sa_exc.DBAPIError:
                    cands = []
                except Exception:
                    cands = []

            # 3) Pure vector fallback
            if not cands and qe is not None:
                try:
                    cands = _candidate_rows(
                        conn, qe, topN,
                        policy_bias=(foa_hint is None),
                    )
                except Exception:
                    cands = []

            # 4) Keyword-only fallback over policy pages (no embeddings required)
            if not cands:
                try:
                    cands = _keyword_fallback_rows(conn, query, k=k)
                except Exception:
                    cands = []

            if not cands:
                return []

            # If embeddings present, apply MMR; else return as-is
            if qe is not None and all(c.get("emb") for c in cands):
                vecs = [c["emb"] for c in cands]
                keep = mmr_rerank(qe, vecs, k=k, lambda_mult=0.7)
                return [cands[i] for i in keep]
            else:
                return cands[:k]
    except Exception:
        return []

def build_context(cands):
    parts, cites = [], []
    for i, r in enumerate(cands, start=1):
        parts.append(f"[{i}] {r['title']}\nURL: {r['url']}\nExcerpt:\n{r['content']}\n")
        cites.append({"id": i, "title": r["title"], "url": r["url"]})
    return "\n\n---\n\n".join(parts)[:16000], cites

class AskRagResponse(BaseModel):
    answer: str
    sources: list

SYSTEM_PROMPT_RAG = (
    "You are an NIH grants application assistant. Use ONLY the provided context to answer. "
    "If the answer depends on a specific FOA/NOFO, state that those instructions supersede general guidance. "
    'Cite sources inline like [1], [2]. Keep answers concise.'
)

@app.post("/ask_rag", response_model=AskRagResponse)
def ask_rag(req: AskRequest):
    q = req.question.strip()
    m = FOA_RE.search(q) or NOTICE_RE.search(q)
    foa_hint = m.group(0).upper() if m else None

    meta_block = ""
    if foa_hint:
        meta = get_foa_meta(foa_hint)
        if meta:
            kd = meta.get("key_dates", {}) or {}
            meta_lines = [
                f"FOA Number: {meta.get('foa_number')}",
                f"Title: {meta.get('title')}",
                f"URL: {meta.get('url')}",
                f"Activity Codes: {', '.join(meta.get('activity_codes') or []) or '—'}",
                f"Participating ICs: {', '.join(meta.get('participating_ics') or []) or '—'}",
                f"Clinical Trial: {meta.get('clinical_trial') or '—'}",
                f"Open Date: {kd.get('open_date','—')}",
                f"LOI Due: {kd.get('loi_due','—')}",
                f"Application Due Date(s): {kd.get('due_dates','—')}",
                f"Expiration Date: {kd.get('expiration','—')}",
            ]
            meta_block = "FOA Metadata:\n" + "\n".join(meta_lines) + "\n\n---\n\n"

    rows = retrieve_top_chunks(q, k=6, foa_hint=foa_hint)
    if not rows:
        return AskRagResponse(
            answer="No relevant NIH pages found in the index yet (or the database is not ready).",
            sources=[]
        )

    context, cites = build_context(rows)
    user_msg = f"User question:\n{q}\n\nContext:\n{meta_block}{context}\n\nReturn a concise, policy-accurate answer with inline [#] citations."

    for attempt in range(5):
        try:
            chat = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":SYSTEM_PROMPT_RAG},
                    {"role":"user","content":user_msg}
                ],
                temperature=0.1,
            )
            return AskRagResponse(answer=chat.choices[0].message.content.strip(), sources=cites)
        except Exception as e:
            msg = str(e)
            if attempt < 4 and any(x in msg for x in ["RateLimit","429","ServiceUnavailable","Timeout"]):
                time.sleep((0.5*(2**attempt)) + random.uniform(0,0.25))
                continue
            raise HTTPException(status_code=503, detail=f"LLM error: {type(e).__name__}")

# ---------------- Admin: stats ----------------
@app.get("/admin/stats", dependencies=[Depends(require_admin)])
def admin_stats():
    with engine.begin() as conn:
        doc_count = conn.execute(text("SELECT count(*) FROM documents")).scalar_one()
        chunk_count = conn.execute(text("SELECT count(*) FROM chunks")).scalar_one()
        emb_count = conn.execute(text("SELECT count(*) FROM embeddings")).scalar_one()
        foa_count = conn.execute(text("SELECT count(*) FROM foas")).scalar_one()
        last_doc = conn.execute(text("SELECT max(fetched_at) FROM documents")).scalar_one()
        last_foa = conn.execute(text("SELECT max(last_seen) FROM foas")).scalar_one()
    def to_iso(dt):
        if not dt: return None
        if isinstance(dt, (datetime.datetime, datetime.date)): return dt.isoformat()
        return str(dt)
    return {
        "documents": doc_count,
        "chunks": chunk_count,
        "embeddings": emb_count,
        "foas": foa_count,
        "last_document_fetch": to_iso(last_doc),
        "last_foa_seen": to_iso(last_foa),
    }

# ---------------- Admin: recent changes ----------------
@app.get("/admin/changes", dependencies=[Depends(require_admin)])
def admin_changes(limit: int = 50):
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, kind, url, title, change_summary, seen_at
            FROM changes
            ORDER BY seen_at DESC
            LIMIT :n
        """), {"n": limit}).fetchall()
    out = []
    for r in rows:
        seen_iso = r.seen_at.isoformat() if hasattr(r.seen_at, "isoformat") else str(r.seen_at)
        out.append({
            "id": r.id,
            "kind": r.kind,
            "url": r.url,
            "title": r.title,
            "summary": r.change_summary,
            "seen_at": seen_iso,
        })
    return out

# ---------------- Admin: on-demand ingest/refresh ----------------
def guess_foa_url(foa_number: str):
    """Return likely NIH Guide URL(s) for a given FOA/Notice number."""
    n = foa_number.upper()
    if n.startswith("RFA-"):
        return [f"https://grants.nih.gov/grants/guide/rfa-files/{n}.htm",
                f"https://grants.nih.gov/grants/guide/rfa-files/{n}.html"]
    if n.startswith("PA-"):
        return [f"https://grants.nih.gov/grants/guide/pa-files/{n}.htm",
                f"https://grants.nih.gov/grants/guide/pa-files/{n}.html"]
    if n.startswith("PAR-"):
        return [f"https://grants.nih.gov/grants/guide/par-files/{n}.htm",
                f"https://grants.nih.gov/grants/guide/par-files/{n}.html"]
    if n.startswith("NOT-"):
        return [f"https://grants.nih.gov/grants/guide/notice-files/{n}.htm",
                f"https://grants.nih.gov/grants/guide/notice-files/{n}.html"]
    return []

class RefreshResponse(BaseModel):
    ok: bool
    tried: list
    message: str = ""

@app.post("/admin/refresh_foa/{foa_number}", response_model=RefreshResponse, dependencies=[Depends(require_admin)])
def refresh_foa(foa_number: str):
    tried = []
    last_err = None
    try:
        from ingest_basic import ingest_url  # lazy import
    except Exception as e:
        return RefreshResponse(ok=False, tried=[], message=f"ingest module not available: {e}")

    for url in guess_foa_url(foa_number):
        tried.append(url)
        try:
            ingest_url(url, visited=set())
            return RefreshResponse(ok=True, tried=tried, message="FOA ingested/refreshed.")
        except Exception as e:
            last_err = str(e)
            continue

    return RefreshResponse(ok=False, tried=tried, message=f"Could not ingest any candidate URL. Last error: {last_err or 'n/a'}")

@app.post("/admin/reindex_url", response_model=RefreshResponse, dependencies=[Depends(require_admin)])
def reindex_url(url: str = Query(..., description="Absolute HTTP(S) URL to ingest or refresh")):
    try:
        from ingest_basic import ingest_url  # lazy import
    except Exception as e:
        return RefreshResponse(ok=False, tried=[url], message=f"ingest module not available: {e}")

    try:
        ingest_url(url, visited=set())
        return RefreshResponse(ok=True, tried=[url], message="URL ingested/refreshed.")
    except Exception as e:
        return RefreshResponse(ok=False, tried=[url], message=str(e))

# ---------------- Admin: retrieval diagnose ----------------
class DiagnoseRequest(BaseModel):
    question: str
    k: int = 6

@app.post("/admin/rag_diagnose", dependencies=[Depends(require_admin)])
def rag_diagnose(req: DiagnoseRequest):
    """
    Returns the top-k rows the retriever sees (title/url/first 240 chars).
    Helps debug when /ask_rag says 'no relevant pages'.
    """
    q = req.question.strip()
    m = FOA_RE.search(q) or NOTICE_RE.search(q)
    foa_hint = m.group(0).upper() if m else None
    rows = retrieve_top_chunks(q, k=max(3, min(req.k, 12)), foa_hint=foa_hint)
    out = []
    for r in rows:
        out.append({
            "title": r["title"],
            "url": r["url"],
            "preview": (r["content"] or "")[:240]
        })
    return {"count": len(out), "items": out}
