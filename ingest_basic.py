# ingest_basic.py
import os
import io
import re
import json
import time
import random
import hashlib
import datetime
from typing import List, Tuple, Dict
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import sqlalchemy as sa
from sqlalchemy import text
from dotenv import load_dotenv
import tiktoken
from openai import OpenAI
from pgvector.psycopg import register_vector, Vector
from pdfminer.high_level import extract_text as pdf_extract_text

# ---------------- Env & clients ----------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not DATABASE_URL or not OPENAI_API_KEY:
    raise RuntimeError("DATABASE_URL or OPENAI_API_KEY missing in .env")

engine = sa.create_engine(DATABASE_URL, future=True)
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- Seeds (core coverage) ----------------
SEED_URLS = [
    # Application Guide & structure
    "https://grants.nih.gov/grants-process/write-application/how-to-apply-application-guide",
    "https://grants.nih.gov/grants/how-to-apply-application-guide/forms-i/general/g.100-how-to-use-the-application-instructions.htm",

    # Formatting & page limits
    "https://grants.nih.gov/grants-process/write-application/how-to-apply-application-guide/format-attachments",
    "https://grants.nih.gov/grants-process/write-application/how-to-apply-application-guide/page-limits",

    # Biosketch (non-fellowship + fellowship)
    "https://grants.nih.gov/grants-process/write-application/forms-directory/non-fellowship-biographical-sketch-format",
    "https://grants.nih.gov/grants-process/write-application/forms-directory/fellowship-biographical-sketch-format",

    # Other Support
    "https://grants.nih.gov/grants-process/write-application/forms-directory/other-support",
    "https://grants.nih.gov/grants-process/write-application/forms-directory/other-support-format-page",

    # Data Management & Sharing
    "https://grants.nih.gov/policy-and-compliance/policy-topics/sharing-policies/dms/policy-overview",
    "https://grants.nih.gov/policy-and-compliance/policy-topics/sharing-policies/dms/writing-dms-plan",
    "https://grants.nih.gov/grants-process/write-application/forms-directory/data-management-and-sharing-plan-format-page",
    "https://grants.nih.gov/policy-and-compliance/policy-topics/sharing-policies/dms/budgeting-for-data-management-sharing",

    # Human Subjects & Clinical Trials
    "https://grants.nih.gov/grants/how-to-apply-application-guide/forms-i/general/g.500-phs-human-subjects-and-clinical-trials-information.htm",
    "https://grants.nih.gov/policy-and-compliance/policy-topics/clinical-trials",
    "https://grants.nih.gov/policy-and-compliance/policy-topics/clinical-trials/definition",
    "https://grants.nih.gov/policy-and-compliance/policy-topics/clinical-trials/case-studies",
    "https://grants.nih.gov/policy-and-compliance/policy-topics/human-subjects/pre-and-post-award-processes/considerations",

    # Budgets
    "https://grants.nih.gov/grants-process/write-application/advice-on-application-sections/develop-your-budget",
    "https://grants.nih.gov/grants/how-to-apply-application-guide/forms-i/general/g.300-r%26r-budget-form.htm",
    "https://grants.nih.gov/grants/how-to-apply-application-guide/forms-i/general/g.320-phs-398-modular-budget-form.htm",

    # Letters of support & glossary
    "https://grants.nih.gov/grants-process/write-application/advice-on-application-sections",
    "https://grants.nih.gov/grants-process/write-application/advice-on-application-sections/reference-letters",
    "https://grants.nih.gov/grants/glossary.htm",
]

ALLOWED_PDF_DOMAINS = {"grants.nih.gov", "nih.gov"}
MAX_PDFS_PER_PAGE = 20

CHUNK_TARGET_TOKENS = 700
OVERLAP_TOKENS = 80
enc = tiktoken.get_encoding("cl100k_base")

USER_AGENT = "NIH-Grant-Chatbot/1.0 (+yourdomain.com/contact)"
HEADERS = {"User-Agent": USER_AGENT}

# ---------------- FOA / Notice detection ----------------
# Accept both .htm and .html, case-insensitive
FOA_HTML_RE = re.compile(
    r"/grants/guide/(?:rfa|pa|par)-files/(?:RFA|PA|PAR)-[A-Z0-9\-]+\.htm(l)?$",
    re.IGNORECASE,
)
NOTICE_HTML_RE = re.compile(
    r"/grants/guide/notice-files/NOT-[A-Z0-9\-]+\.htm(l)?$",
    re.IGNORECASE,
)

# Numbers in text or URL
FOA_NUM_RE = re.compile(r"\b(RFA|PA|PAR)-[A-Z]{2,}-\d{2}-\d{3}[A-Z]?\b", re.IGNORECASE)
NOTICE_NUM_RE = re.compile(r"\bNOT-[A-Z]{2,}-\d{2}-\d{3}\b", re.IGNORECASE)

# Extract from URL directly (preferred)
FOA_NUM_FROM_URL_RE = re.compile(r"(RFA|PA|PAR)-[A-Z]{2,}-\d{2}-\d{3}[A-Z]?", re.IGNORECASE)
NOTICE_NUM_FROM_URL_RE = re.compile(r"(NOT-[A-Z]{2,}-\d{2}-\d{3})", re.IGNORECASE)

# ---------------- Utilities ----------------
def normalize_ws(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def checksum(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def chunk_text(text: str, target=CHUNK_TARGET_TOKENS, overlap=OVERLAP_TOKENS) -> List[str]:
    toks = enc.encode(text)
    chunks, i = [], 0
    while i < len(toks):
        j = min(i + target, len(toks))
        chunks.append(enc.decode(toks[i:j]))
        i = j - overlap if j - overlap > i else j
    return chunks

def embed_with_retry(texts: List[str]) -> List[List[float]]:
    # 1536-dim model to match DB schema
    max_tries = 5
    for attempt in range(max_tries):
        try:
            resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            msg = str(e)
            transient = any(t in msg for t in ("RateLimit", "429", "ServiceUnavailable", "Timeout"))
            if attempt < max_tries - 1 and transient:
                wait = (0.5 * (2 ** attempt)) + random.uniform(0, 0.25)
                print(f"   transient embedding error: {msg} — retrying in {wait:.2f}s")
                time.sleep(wait)
                continue
            raise

def fetch_and_parse(url: str) -> Tuple[str, str, List[str], str]:
    """
    Returns (title, text, pdf_links, content_type) where content_type is 'pdf' or 'html'.
    For HTML, also returns discovered NIH PDF links.
    """
    r = requests.get(url, timeout=45, headers=HEADERS)
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()
    is_pdf = ("application/pdf" in ctype) or r.content.startswith(b"%PDF")

    if is_pdf:
        # ----- PDF -----
        try:
            text = pdf_extract_text(io.BytesIO(r.content)) or ""
        except Exception as e:
            raise RuntimeError(f"PDF parse failed: {e}")
        fn = urlparse(url).path.rsplit("/", 1)[-1] or "document.pdf"
        title = fn
        text = normalize_ws(text)
        return title, text, [], "pdf"

    # ----- HTML -----
    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    title = (soup.title.string.strip() if soup.title and soup.title.string else url)
    parts = [
        t.get_text(" ", strip=True)
        for t in soup.find_all(["h1", "h2", "h3", "p", "li", "table", "thead", "tbody", "tr", "td"])
    ]
    text = normalize_ws("\n".join([p for p in parts if p]))

    # Discover NIH-hosted PDF links on the page
    pdf_links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href:
            continue
        abs_url = urljoin(url, href)
        if not abs_url.lower().endswith(".pdf"):
            continue
        netloc = urlparse(abs_url).netloc.lower()
        if any(netloc.endswith(d) for d in ALLOWED_PDF_DOMAINS):
            pdf_links.append(abs_url)

    # de-dupe and cap
    seen, filtered = set(), []
    for p in pdf_links:
        if p not in seen:
            filtered.append(p)
            seen.add(p)
        if len(filtered) >= MAX_PDFS_PER_PAGE:
            break

    return title, text, filtered, "html"

# ---------------- DB upserts ----------------
def upsert_document(conn, url: str, title: str, body: str):
    csum = checksum(body)
    row = conn.execute(text("SELECT id, checksum FROM documents WHERE url=:u"), {"u": url}).fetchone()
    now = datetime.datetime.utcnow()

    if row and row.checksum == csum:
        return row.id, False

    if row:
        doc_id = row.id
        conn.execute(
            text(
                """
                UPDATE documents
                SET title=:t, fetched_at=:f, checksum=:c
                WHERE id=:id
                """
            ),
            {"t": title, "f": now, "c": csum, "id": doc_id},
        )
        conn.execute(text("DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id=:id)"), {"id": doc_id})
        conn.execute(text("DELETE FROM chunks WHERE doc_id=:id"), {"id": doc_id})
        return doc_id, True

    res = conn.execute(
        text(
            """
            INSERT INTO documents (url, title, fetched_at, checksum)
            VALUES (:u, :t, :f, :c)
            RETURNING id
            """
        ),
        {"u": url, "t": title, "f": now, "c": csum},
    )
    return res.scalar_one(), True

def upsert_foa(conn, foa: Dict):
    """
    Upsert into foas table. key_dates_json is cast using CAST(:param AS JSONB)
    to avoid driver-specific placeholder issues.
    """
    by_number = foa.get("foa_number")
    if by_number:
        existing = conn.execute(text("SELECT id FROM foas WHERE foa_number=:n"), {"n": by_number}).fetchone()
    else:
        existing = conn.execute(text("SELECT id FROM foas WHERE url=:u"), {"u": foa["url"]}).fetchone()

    fields = {
        "foa_number": foa.get("foa_number"),
        "foa_type": foa.get("foa_type"),
        "title": foa.get("title"),
        "url": foa.get("url"),
        "activity_codes": foa.get("activity_codes"),
        "participating_ics": foa.get("participating_ics"),
        "clinical_trial": foa.get("clinical_trial"),
        "key_dates_json": json.dumps(foa.get("key_dates") or {}),
    }

    if existing:
        conn.execute(
            text(
                """
                UPDATE foas SET
                  foa_number        = COALESCE(:foa_number, foa_number),
                  foa_type          = :foa_type,
                  title             = :title,
                  url               = :url,
                  activity_codes    = :activity_codes,
                  participating_ics = :participating_ics,
                  clinical_trial    = :clinical_trial,
                  key_dates_json    = CAST(:key_dates_json AS JSONB),
                  last_seen         = now()
                WHERE id = :id
                """
            ),
            {**fields, "id": existing.id},
        )
    else:
        conn.execute(
            text(
                """
                INSERT INTO foas
                  (foa_number, foa_type, title, url, activity_codes, participating_ics, clinical_trial, key_dates_json)
                VALUES
                  (:foa_number, :foa_type, :title, :url, :activity_codes, :participating_ics, :clinical_trial, CAST(:key_dates_json AS JSONB))
                """
            ),
            fields,
        )

# ---------------- FOA parsing helpers ----------------
def parse_list_field(text_block: str) -> List[str]:
    parts = re.split(r"[;,]\s*|\s{2,}", text_block)
    return [p.strip() for p in parts if p.strip()]

def extract_foa_fields_from_text(url: str, title: str, text_body: str) -> Dict:
    """
    Prefer FOA/Notice number from URL; fall back to body text.
    """
    d: Dict = {"activity_codes": None, "participating_ics": None, "clinical_trial": None, "key_dates": {}}

    # Prefer number from URL
    if FOA_HTML_RE.search(url):
        m = FOA_NUM_FROM_URL_RE.search(url)
        if m: d["foa_number"] = m.group(0).upper(); d["foa_type"] = d["foa_number"].split("-")[0]
    elif NOTICE_HTML_RE.search(url):
        m = NOTICE_NUM_FROM_URL_RE.search(url)
        if m: d["foa_number"] = m.group(0).upper(); d["foa_type"] = "NOT"

    # Fall back to body text if still missing
    if not d.get("foa_number"):
        m = FOA_NUM_RE.search(text_body) or NOTICE_NUM_RE.search(text_body)
        if m:
            d["foa_number"] = m.group(0).upper()
            d["foa_type"] = d["foa_number"].split("-")[0] if d["foa_number"].startswith(("RFA","PA","PAR")) else "NOT"

    # Title from page if present
    d["title"] = title or d.get("title")

    # Activity Codes
    m = re.search(r"Activity Code\(s\)\s*[:\-]\s*(.+)", text_body, re.IGNORECASE)
    if m: d["activity_codes"] = parse_list_field(m.group(1))

    # Participating Components / ICs
    m = re.search(r"(Participating Organization\(s\)|Components|Participating ICs?)\s*[:\-]\s*(.+)", text_body, re.IGNORECASE)
    if m: d["participating_ics"] = parse_list_field(m.group(2))

    # Clinical Trial
    m = re.search(r"Clinical Trial(?:\s*:\s*|\s*)(Required|Optional|Not Allowed)", text_body, re.IGNORECASE)
    if m: d["clinical_trial"] = m.group(1).title()

    # Key Dates
    kd = {}
    pairs = [
        ("open_date", r"Open Date(?:.*?):\s*(.+)"),
        ("loi_due", r"Letter of Intent(?:.*)Due Date\(s\)\s*:\s*(.+)"),
        ("due_dates", r"Application Due Date\(s\)\s*:\s*(.+)"),
        ("expiration", r"Expiration Date\s*:\s*(.+)"),
    ]
    for key, pat in pairs:
        mm = re.search(pat, text_body, re.IGNORECASE)
        if mm: kd[key] = mm.group(1).strip()
    if kd: d["key_dates"] = kd

    return d

def is_foa_page(url: str) -> bool:
    return bool(FOA_HTML_RE.search(urlparse(url).path))

def is_notice_page(url: str) -> bool:
    return bool(NOTICE_HTML_RE.search(urlparse(url).path))

# ---------------- Save chunks & embeddings ----------------
def save_chunks_and_embeddings(conn, doc_id: int, chunks: List[str], embeddings: List[List[float]]):
    raw = conn.connection.driver_connection
    register_vector(raw)
    for idx, (content, emb) in enumerate(zip(chunks, embeddings), start=1):
        res = conn.execute(
            text(
                """
                INSERT INTO chunks (doc_id, ordinal, heading_path, content, token_count)
                VALUES (:doc, :ord, :hp, :ct, :tk)
                RETURNING id
                """
            ),
            {"doc": doc_id, "ord": idx, "hp": None, "ct": content, "tk": len(enc.encode(content))},
        )
        chunk_id = res.scalar_one()
        conn.execute(
            text("INSERT INTO embeddings (chunk_id, embedding) VALUES (:cid, :emb)"),
            {"cid": chunk_id, "emb": Vector(emb)},
        )

# ---------------- Main ingest function ----------------
def ingest_url(url: str, visited: set):
    if url in visited:
        return
    visited.add(url)

    print(f"🌐 Fetching: {url}")
    title, text_body, pdf_links, ctype = fetch_and_parse(url)
    if not text_body.strip():
        print("   (Empty page)")
        return

    with engine.begin() as conn:
        doc_id, changed = upsert_document(conn, url, title, text_body)

        # FOA/Notice metadata (HTML pages)
        if ctype == "html" and (is_foa_page(url) or is_notice_page(url)):
            foa = extract_foa_fields_from_text(url, title, text_body)
            foa["url"] = url
            upsert_foa(conn, foa)
            print(f"   🧾 FOA parsed: {foa.get('foa_number')} — {foa.get('title')}")

        if changed:
            chunks = chunk_text(text_body)
            print(f"   Split into {len(chunks)} chunks")
            embeddings = embed_with_retry(chunks)
            save_chunks_and_embeddings(conn, doc_id, chunks, embeddings)
            print(f"   ✅ Ingested {len(chunks)} chunks for: {title}")
        else:
            print("   No change — skipping re-embed.")

    # Auto-ingest NIH PDFs discovered on HTML pages
    for pdf_url in pdf_links:
        try:
            ingest_url(pdf_url, visited)
        except Exception as e:
            print(f"   ❌ Error ingesting PDF {pdf_url}: {e}")

# ---------------- Script entry ----------------
if __name__ == "__main__":
    visited = set()
    for u in SEED_URLS:
        try:
            ingest_url(u, visited)
        except Exception as e:
            print(f"   ❌ Error ingesting {u}: {e}")
    print("Done.")
