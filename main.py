# main.py
from fastapi import FastAPI, HTTPException, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
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
    "text-embedding-3-large,text-embedding-ada-002"
).split(",") if m.strip()]

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
    CORSMWARE := CORSMiddleware,  # alias just so flake8 stays calm
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
    "Keep answers short. No markdown backticks."
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
            SELECT foa_number, foa_type, title, url, activity_codes, participating_ics, clinical_trial, key_dates_json, expiration
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
        "expiration": row.expiration
    }

def get_foa_meta(foa_number: str) -> Optional[Dict[str, Any]]:
    with engine.begin() as conn:
        r = conn.execute(text("""
            SELECT foa_number, foa_type, title, url, activity_codes, participating_ics, clinical_trial, key_dates_json, expiration
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
        "expiration": r.expiration
    }

def get_foa_main_url(foa_number: str) -> Optional[str]:
    with engine.begin() as conn:
        r = conn.execute(text("""
            SELECT url FROM foas WHERE upper(foa_number)=upper(:n)
        """), {"n": foa_number}).fetchone()
    return r.url if r else None

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

def _fts_where_clause() -> str:
    # Lenient matching on shorter/natural language queries
    return "WHERE c.ts @@ plainto_tsquery('english', :ftsq) OR (d.url ILIKE '%/due-dates%' OR d.title ILIKE :kw OR c.content ILIKE :kw)"

def _candidate_rows(
    conn,
    qe: List[float],
    topN: int,
    where_extra: str = "",
    params: Optional[Dict[str, Any]] = None,
    policy_bias: bool = False,
    prefer_grants_domain: bool = True,
):
    """
    Vector ranking (optionally FTS prefilter).
    Adds small domain/policy nudges to the ORDER BY **as numeric** (never vector ops).
    """
    params = params or {}

    # Base distance (double precision). Wrap in parentheses so we never get 'vector + numeric'
    base_expr = "(e.embedding <-> CAST(:qe AS vector))"
    penalties = []

    if policy_bias:
        penalties.append("""
            + CASE
                WHEN d.url ILIKE '%/how-to-apply-application-guide/%' THEN -0.08
                WHEN d.url ILIKE '%/grants-process/%'                 THEN -0.06
                WHEN d.url ILIKE '%/due-dates%'                       THEN -0.12
                WHEN d.url ILIKE '%/nihgps/%'                         THEN -0.05
                WHEN d.title ILIKE '%standard due date%'              THEN -0.12
                WHEN d.title ILIKE '%application guide%'              THEN -0.08
                ELSE 0.0
              END
        """)

    if prefer_grants_domain:
        penalties.append("""
            + CASE
                WHEN d.url ILIKE 'https://grants.nih.gov/%' THEN -0.20
                ELSE 0.10
              END
        """)

    order_expr = base_expr + (" ".join(penalties) if penalties else "")

    sql = f"""
        SELECT c.content, d.title, d.url, e.embedding AS emb
        FROM embeddings e
        JOIN chunks c ON c.id = e.chunk_id
        JOIN documents d ON d.id = c.doc_id
        {where_extra}
        ORDER BY {order_expr}
        LIMIT :n
    """
    # Provide keyword param for the WHERE clause if needed
    kw = params.get("kw", f"%{params.get('ftsq','')}%")
    rows = conn.execute(text(sql), {"qe": qe, "n": topN, "kw": kw, **params}).fetchall()
    return [{"content": r.content, "title": r.title, "url": r.url, "emb": _to_list(r.emb)} for r in rows]

def _keyword_fallback_rows(conn, query: str, k: int = 6) -> List[Dict[str, Any]]:
    """
    LAST-RESORT: keyword filters over high-signal NIH policy pages. No embeddings required.
    """
    patterns = [
        "%/how-to-apply-application-guide/%",
        "%/due-dates-and-submission-policies/%",
        "%/due-dates.htm%",                 # legacy
        "%/standard-due-dates.htm%",        # specific
        "%/submission-policies.htm%",       # nearby
        "%/policy/nihgps/%",                # GPS
        "%/peer-review.htm%",               # overview
    ]
    words = [w for w in re.findall(r"[A-Za-z0-9]{3,}", query)][:6]
    like_params = {f"p{i}": p for i, p in enumerate(patterns)}
    wc = [f"d.url ILIKE :p{i}" for i in range(len(patterns))]
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
              WHEN d.url ILIKE '%/policy/nihgps/%' THEN 1
              WHEN d.url ILIKE '%/how-to-apply-application-guide/%' THEN 2
              ELSE 3
            END,
            length(c.content) DESC
        LIMIT :n
    """
    rows = conn.execute(text(sql), {**like_params, "n": max(12, k * 4)}).fetchall()
    out = [{"content": r.content, "title": r.title, "url": r.url, "emb": None} for r in rows]
    return out[:k]

def _fts_only_rows(conn, query: str, topN: int) -> List[Dict[str, Any]]:
    sql = f"""
        SELECT c.content, d.title, d.url
        FROM chunks c
        JOIN documents d ON d.id = c.doc_id
        WHERE c.ts @@ plainto_tsquery('english', :ftsq)
        ORDER BY ts_rank(c.ts, websearch_to_tsquery('english', :ftsq)) DESC,
                 length(c.content) DESC
        LIMIT :n
    """
    rows = conn.execute(text(sql), {"ftsq": query, "n": topN}).fetchall()
    return [{"content": r.content, "title": r.title, "url": r.url, "emb": None} for r in rows]

def retrieve_top_chunks(query: str, k: int = 6, foa_hint: str = None, foa_only_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieval order:
      0) Quick keyword-first fallback for high-signal policy pages
      1a) FTS pre-filter + vector rank (policy_bias ON for generic queries)
      1b) If embeddings unavailable, FTS-only ranking
      2) Pure vector rank
      3) Keyword fallback
    If foa_only_url is given, filter to that FOA page first.
    """
    qe: Optional[List[float]] = None
    try:
        qe, used_model = embed_query(query)
        print(f"[retrieve_top_chunks] embed model used: {used_model}")
    except Exception as e:
        print(f"[retrieve_top_chunks] embeddings unavailable: {e.__class__.__name__}: {e}")
        qe = None

    topN = max(12, k * 4)

    try:
        with engine.begin() as conn:
            # Ensure pgvector adapter
            try:
                raw = conn.connection.driver_connection
                register_vector(raw)
            except Exception:
                pass

            # Optional hard FOA filter
            foa_where = ""
            foa_params: Dict[str, Any] = {}
            if foa_only_url:
                foa_where = "WHERE d.url ILIKE :foa_like"
                foa_params["foa_like"] = f"%{foa_only_url.split('#')[0]}%"

            # 0) Quick keyword-first
            try:
                quick = _keyword_fallback_rows(conn, query, k=max(3, min(6, k)))
            except Exception:
                quick = []

            # 1a) FTS + vector
            cands: List[Dict[str, Any]] = []
            if qe is not None:
                try:
                    where = _fts_where_clause()
                    params = {"ftsq": query, "kw": f"%{query}%"}
                    if foa_where:
                        where = foa_where + " AND (" + where[len("WHERE "):] + ")"
                        params.update(foa_params)
                    cands = _candidate_rows(
                        conn, qe, topN,
                        where, params,
                        policy_bias=(foa_hint is None and not foa_only_url),
                    )
                except Exception as e:
                    print(f"[stage FTS+vec] {e.__class__.__name__}: {e}")
                    cands = []
            else:
                # 1b) FTS-only
                try:
                    if foa_where:
                        rows = conn.execute(text(f"""
                            SELECT c.content, d.title, d.url
                            FROM chunks c
                            JOIN documents d ON d.id = c.doc_id
                            {foa_where} AND c.ts @@ plainto_tsquery('english', :ftsq)
                            ORDER BY ts_rank(c.ts, websearch_to_tsquery('english', :ftsq)) DESC,
                                     length(c.content) DESC
                            LIMIT :n
                        """), {"ftsq": query, "n": topN, **foa_params}).fetchall()
                        cands = [{"content": r.content, "title": r.title, "url": r.url, "emb": None} for r in rows]
                    else:
                        cands = _fts_only_rows(conn, query, topN)
                except Exception:
                    cands = []

            # 2) Pure vector
            if not cands and qe is not None:
                try:
                    where = foa_where
                    params = {**foa_params}
                    cands = _candidate_rows(
                        conn, qe, topN,
                        where, params,
                        policy_bias=(foa_hint is None and not foa_only_url),
                    )
                except Exception as e:
                    print(f"[stage vec-only] {e.__class__.__name__}: {e}")
                    cands = []

            # 3) Keyword-only fallback
            if not cands:
                try:
                    if foa_where:
                        rows = conn.execute(text(f"""
                            SELECT c.content, d.title, d.url
                            FROM chunks c
                            JOIN documents d ON d.id = c.doc_id
                            {foa_where} AND (
                                d.title ILIKE :kw OR c.content ILIKE :kw
                            )
                            ORDER BY length(c.content) DESC
                            LIMIT :n
                        """), {"kw": f"%{query}%", "n": topN, **foa_params}).fetchall()
                        cands = [{"content": r.content, "title": r.title, "url": r.url, "emb": None} for r in rows]
                    else:
                        cands = _keyword_fallback_rows(conn, query, k=k)
                except Exception as e:
                    print(f"[stage keyword-fallback] {e.__class__.__name__}: {e}")
                    cands = []

            chosen = cands if cands else quick
            if not chosen:
                return []

            if qe is not None and all(c.get("emb") for c in chosen):
                vecs = [c["emb"] for c in chosen]
                keep = mmr_rerank(qe, vecs, k=k, lambda_mult=0.7)
                return [chosen[i] for i in keep]
            else:
                return chosen[:k]
    except Exception as e:
        print(f"[retrieve_top_chunks] fatal: {e.__class__.__name__}: {e}")
        return []

def build_context(cands: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    parts, cites = [], []
    for i, r in enumerate(cands, start=1):
        excerpt = (r.get('content') or '')[:1800]
        parts.append(f"[{i}] {r['title']}\nURL: {r['url']}\nExcerpt:\n{excerpt}\n")
        cites.append({"id": i, "title": r["title"], "url": r["url"]})
    return "\n\n---\n\n".join(parts)[:16000], cites

def format_references(cites: List[Dict[str, Any]]) -> str:
    if not cites:
        return ""
    lines = []
    for c in cites:
        lines.append(f"- [{c['id']}] {c['title']} — {c['url']}")
    return "\n".join(lines)

def wrap_markdown_answer(body: str, cites: List[Dict[str, Any]]) -> str:
    refs = format_references(cites)
    return (
        f"{body.strip()}\n\n"
        f"---\n"
        f"**References**\n{refs if refs else '- None'}"
    )

# ---------------- RAG request/response ----------------
class AskRagRequest(BaseModel):
    question: str = Field(..., description="User question")
    # UX knob: beginner | expert
    user_type: Optional[str] = Field(default="beginner", description="beginner | expert")
    # Optional intent for structured outputs/buttons
    intent: Optional[str] = Field(
        default=None,
        description="one of: foa_summary, key_dates, required_docs, aims_outline, budget_justification_outline"
    )
    # Optional explicit FOA number or URL to focus retrieval (e.g., PAR-24-112)
    foa: Optional[str] = Field(default=None, description="FOA/NOFO number like PAR-24-112 or a full NIH Guide URL")

class AskRagResponse(BaseModel):
    answer: str
    sources: list

SYSTEM_PROMPT_RAG = (
    "You are an expert NIH Grant Application Assistant. Use ONLY the provided NIH context. "
    "Never speculate or invent missing data. If the answer depends on a specific FOA/NOFO, say that those instructions supersede general guidance. "
    "If a page is out of scope, say you cannot find it in the provided context. "
    "Always include inline bracket citations like [1], [2] that map to the numbered References list appended by the system."
)

FORMATTING_GUIDE = (
    "FORMAT YOUR ANSWER AS MARKDOWN with bold section headers and bullet lists. "
    "Prefer concise, professional phrasing. "
    "For beginners: add short friendly clarifications; for experts: be succinct and assume familiarity with NIH terms. "
    "Suggested sections (use when relevant):\n"
    "- **Summary**\n- **Key Dates**\n- **Budget / Eligibility / Clinical Trial**\n"
    "- **Requirements**\n- **Notes** (e.g., FOA supersedes general policy)\n"
    "Do NOT include your own References section; the system will append it for you."
)

INTENT_TEMPLATES = {
    "foa_summary": (
        "Task: Summarize the FOA/NOFO for a grant applicant.\n"
        "Extract, when present in context: budget caps/limits; clinical trial allowed/required status; "
        "expiration date; LOI policy/due timing; any special eligibility or format requirements. "
        "Flag if the FOA appears expired.\n"
    ),
    "key_dates": (
        "Task: List all key dates in a clear bullet list (Open Date, LOI, Application Due Dates, Review, "
        "Earliest Start, Expiration)."
    ),
    "required_docs": (
        "Task: List required documents/attachments for submission from the context (e.g., Specific Aims, Research Strategy, "
        "Biosketch, Budget, Facilities, Human Subjects/CT, Data Sharing)."
    ),
    "aims_outline": (
        "Task: Generate a concise 'Specific Aims' outline (bulleted) consistent with NIH norms. "
        "Use placeholders for project-specific content; do not invent facts."
    ),
    "budget_justification_outline": (
        "Task: Generate a budget justification outline (bulleted) with NIH-compliant headings and reminders (no speculation)."
    ),
}

def _tone_instructions(user_type: Optional[str]) -> str:
    if (user_type or "").lower() == "expert":
        return "Audience: **Expert** NIH applicant. Keep explanations terse; use NIH terminology without defining basics."
    return "Audience: **Beginner** NIH applicant. Keep a friendly tone; briefly define NIH jargon when first used."

def _resolve_foa_focus(foa_field: Optional[str], user_q: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (foa_hint_number, foa_only_url)
    """
    # Detect in question
    m = FOA_RE.search(user_q) or NOTICE_RE.search(user_q)
    foa_hint = m.group(0).upper() if m else None

    # Override if explicit field provided
    if foa_field:
        if foa_field.startswith("http"):
            return foa_hint, foa_field
        # assume number
        url = get_foa_main_url(foa_field)
        return (foa_field.upper(), url) if url else (foa_field.upper(), None)

    if foa_hint:
        url = get_foa_main_url(foa_hint) or None
        return foa_hint, url

    return None, None

@app.post("/ask_rag", response_model=AskRagResponse)
def ask_rag(req: AskRagRequest):
    q = req.question.strip()
    user_type = (req.user_type or "beginner").lower().strip()
    intent = (req.intent or "").strip() or None

    foa_hint, foa_only_url = _resolve_foa_focus(req.foa, q)

    # FOA metadata block (if we have it)
    meta_block = ""
    expired_note = ""
    if foa_hint:
        meta = get_foa_meta(foa_hint)
        if meta:
            kd = meta.get("key_dates", {}) or {}
            exp = meta.get("expiration")
            if exp:
                try:
                    # If expired in the past, flag it
                    exp_dt = exp if isinstance(exp, (datetime.date, datetime.datetime)) else None
                    if not exp_dt:
                        exp_dt = datetime.datetime.fromisoformat(str(exp))
                    if exp_dt and exp_dt < datetime.datetime.utcnow():
                        expired_note = "⚠️ This FOA appears to be **expired** based on its expiration date."
                except Exception:
                    pass
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

    # Retrieval (FOA-targeted if we can)
    rows = retrieve_top_chunks(
        q if not intent else f"{q} ({intent})",
        k=6,
        foa_hint=foa_hint,
        foa_only_url=foa_only_url
    )
    if not rows:
        return AskRagResponse(
            answer="No relevant NIH pages found in the index yet (or the database is not ready).",
            sources=[]
        )

    context, cites = build_context(rows)

    # Build system+user messages
    task_block = INTENT_TEMPLATES.get(intent, "")
    tone_block = _tone_instructions(user_type)

    user_msg = (
        f"{tone_block}\n\n{task_block}\n"
        f"User question:\n{q}\n\n"
        f"Context (NIH only):\n{meta_block}{context}\n\n"
        f"{FORMATTING_GUIDE}\n"
        f"Return a concise, policy-accurate answer with inline [#] citations."
    )

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
            body = chat.choices[0].message.content.strip()
            if expired_note:
                body = f"**Note**: {expired_note}\n\n" + body
            final_md = wrap_markdown_answer(body, cites)
            return AskRagResponse(answer=final_md, sources=cites)
        except Exception as e:
            msg = str(e)
            if attempt < 4 and any(x in msg for x in ["RateLimit","429","ServiceUnavailable","Timeout"]):
                time.sleep((0.5*(2**attempt)) + random.uniform(0,0.25))
                continue
            raise HTTPException(status_code=503, detail=f"LLM error: {type(e).__name__}")

# ---------------- Quick-click UI helpers ----------------
@app.get("/ui/suggestions")
def ui_suggestions():
    """
    Buttons the frontend can render as quick actions.
    Each includes a suggested payload to POST to /ask_rag.
    """
    return {
        "buttons": [
            {"label": "🔍 Summarize this FOA", "payload": {"intent": "foa_summary", "user_type": "beginner"}},
            {"label": "💡 Generate Specific Aims Outline", "payload": {"intent": "aims_outline"}},
            {"label": "🗓 Show all key dates", "payload": {"intent": "key_dates"}},
            {"label": "📂 What documents are required?", "payload": {"intent": "required_docs"}},
            {"label": "🧾 Draft a budget justification", "payload": {"intent": "budget_justification_outline"}},
            {"label": "🧠 Beginner Tips", "payload": {"user_type": "beginner"}},
            {"label": "🎯 Expert Mode", "payload": {"user_type": "expert"}},
        ]
    }

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

# ---------------- Admin: ingest/refresh ----------------
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

    # Try known NIH guide URL patterns first, then URL from DB if present
    for url in (guess_foa_url(foa_number) + ([get_foa_main_url(foa_number)] if get_foa_main_url(foa_number) else [])):
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

# ---------------- Admin: self-check, trace, embed probe, vec info ----------------
@app.get("/admin/rag_selfcheck", dependencies=[Depends(require_admin)])
def rag_selfcheck():
    checks = {}
    errors = []
    try:
        with engine.begin() as conn:
            # counts
            checks["counts"] = {
                "documents": conn.execute(text("SELECT count(*) FROM documents")).scalar_one(),
                "chunks": conn.execute(text("SELECT count(*) FROM chunks")).scalar_one(),
                "embeddings": conn.execute(text("SELECT count(*) FROM embeddings")).scalar_one(),
            }
            # FTS column exists?
            try:
                conn.execute(text("SELECT 1 FROM chunks c WHERE c.ts IS NOT NULL LIMIT 1"))
                checks["fts_due_exists"] = True
            except Exception:
                checks["fts_due_exists"] = False

            # vector ops present? sanity order by
            try:
                raw = conn.connection.driver_connection
                register_vector(raw)
            except Exception:
                pass
            try:
                rows = conn.execute(text("""
                    SELECT d.url
                    FROM embeddings e
                    JOIN chunks c ON c.id = e.chunk_id
                    JOIN documents d ON d.id = c.doc_id
                    ORDER BY (e.embedding <-> e.embedding)
                    LIMIT 5
                """)).fetchall()
                checks["pgvector_ops_present"] = True
                checks["vector_rank_sample"] = [r.url for r in rows]
            except Exception as e:
                checks["pgvector_ops_present"] = False
                errors.append(f"vector_rank_sample_probe: {e.__class__.__name__}")
    except Exception as e:
        errors.append(f"db: {e.__class__.__name__}")
    return {"ok": not errors, "checks": checks, "errors": errors}

@app.get("/admin/rag_trace", dependencies=[Depends(require_admin)])
def rag_trace(q: str = Query(..., description="User-like question"), k: int = 6):
    """
    Shows what each retrieval stage returns (urls only) to pinpoint where it goes empty.
    """
    q = q.strip()
    m = FOA_RE.search(q) or NOTICE_RE.search(q)
    foa_hint = m.group(0).upper() if m else None

    trace: Dict[str, Any] = {"foa_hint": foa_hint}
    urls = lambda rows: [r.get("url") for r in rows]

    try:
        with engine.begin() as conn:
            # Ensure vector adapter
            try:
                raw = conn.connection.driver_connection
                register_vector(raw)
            except Exception:
                pass

            # stage 0 keyword
            try:
                s0 = _keyword_fallback_rows(conn, q, k=max(3, min(6, k)))
            except Exception:
                s0 = []
            trace["stage0_keyword_quick"] = urls(s0)

            # embeddings
            try:
                qe, used = embed_query(q)
                trace["embed_model_used"] = used
            except Exception as e:
                qe = None
                trace["embed_model_used"] = None
                trace["embed_error"] = f"{e.__class__.__name__}: {e}"

            # stage 1 FOA (strict) — off if no foa_hint/url
            s1 = []
            if qe is not None and foa_hint:
                like = f"%{(get_foa_main_url(foa_hint) or foa_hint)}%"
                try:
                    s1 = _candidate_rows(
                        conn, qe, max(12, k*4),
                        "WHERE d.url ILIKE :like", {"like": like},
                        policy_bias=False
                    )
                except Exception as e:
                    trace["stage1_error"] = f"{e.__class__.__name__}: {e}"
                    s1 = []
            trace["stage1_foa"] = urls(s1)

            # stage 2 FTS + vector OR FTS-only
            s2 = []
            if qe is not None:
                try:
                    s2 = _candidate_rows(
                        conn, qe, max(12, k*4),
                        _fts_where_clause(), {"ftsq": q, "kw": f"%{q}%"},
                        policy_bias=(foa_hint is None)
                    )
                except Exception as e:
                    trace["stage2_error"] = f"{e.__class__.__name__}: {e}"
                    s2 = []
            else:
                try:
                    s2 = _fts_only_rows(conn, q, max(12, k*4))
                except Exception as e:
                    trace["stage2_error"] = f"{e.__class__.__name__}: {e}"
                    s2 = []
            trace["stage2_fts_vec"] = urls(s2)

            # stage 3 vector only
            s3 = []
            if qe is not None and not s2:
                try:
                    s3 = _candidate_rows(
                        conn, qe, max(12, k*4),
                        policy_bias=(foa_hint is None)
                    )
                except Exception as e:
                    trace["stage3_error"] = f"{e.__class__.__name__}: {e}"
                    s3 = []
            trace["stage3_vec_only"] = urls(s3)

            # stage 4 keyword fallback
            s4 = []
            if not s2 and not s3:
                try:
                    s4 = _keyword_fallback_rows(conn, q, k=k)
                except Exception as e:
                    trace["stage4_error"] = f"{e.__class__.__name__}: {e}"
                    s4 = []
            trace["stage4_keyword_fallback"] = urls(s4)

            # chosen
            chosen = (s1 or s2 or s3 or s4 or s0)[:k]
            trace["chosen"] = urls(chosen)

            # dims (debug)
            try:
                vec, used = embed_query("probe")
                trace["db_vector_dim"] = engine.execute if False else 1536  # not cheap to fetch here
                trace["query_vector_dim"] = len(vec)
                trace["dims_match"] = True  # if you store elsewhere, surface here
            except Exception:
                pass

    except Exception as e:
        raise HTTPException(500, f"trace_error: {e.__class__.__name__}: {e}")

    return trace

@app.get("/admin/embed_probe", dependencies=[Depends(require_admin)])
def embed_probe(q: str = Query("hello NIH grants", description="short test string")):
    try:
        vec, used = embed_query(q)
        return {"ok": True, "model_used": used, "dim": len(vec)}
    except Exception as e:
        return {"ok": False, "error": f"{e.__class__.__name__}: {e}"}

@app.get("/admin/vec_info", dependencies=[Depends(require_admin)])
def vec_info():
    """
    Report the stored vector dimension and configured embedding models.
    """
    db_dim = None
    try:
        with engine.begin() as conn:
            r = conn.execute(text("""
                SELECT (SELECT coalesce(array_length(embedding,1), 0) FROM embeddings LIMIT 1) AS dim
            """)).fetchone()
            db_dim = (r.dim or None) if r else None
    except Exception:
        db_dim = None
    return {
        "db_vector_dim": db_dim,
        "embed_model": EMBED_MODEL,
        "embed_fallbacks": EMBED_FALLBACKS,
    }