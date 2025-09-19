# crawl_nih.py
import re
import time
import queue
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# Reuse your ingestion (handles HTML+PDF, FOA parsing, embeddings)
from ingest_basic import ingest_url

# ---------------- Config ----------------
START_URLS = [
    # NIH Guide entry points
    "https://grants.nih.gov/grants/guide/index.html",
    "https://grants.nih.gov/grants/guide/weeklyindex.cfm",
    "https://grants.nih.gov/grants/guide/WeeklyIndexMobile.cfm",

    # Current-year FOAs / Notices search pages
    "https://grants.nih.gov/grants/guide/search_results.htm?year=Current&scope=pa-rfa",
    "https://grants.nih.gov/grants/guide/search_results.htm?year=Current&scope=notices",
]

ALLOWED_DOMAINS = {"grants.nih.gov"}
ALLOWED_SCHEMES = {"https"}

MAX_PAGES_PER_RUN = 300      # safety cap per crawl
MAX_DEPTH = 4                # how far to follow links from each seed
REQUEST_TIMEOUT = 30
CRAWL_DELAY_SEC = 1.0        # be polite

USER_AGENT = "NIH-Grant-Chatbot/1.0 (+yourdomain.com/contact)"
HEADERS = {"User-Agent": USER_AGENT}

# ---------------- URL patterns ----------------
# FOA (RFA/PA/PAR) and Notice pages (.htm OR .html)
FOA_HTML_ALLOW = re.compile(
    r"/grants/guide/(?:rfa|pa|par)-files/(?:RFA|PA|PAR)-[A-Z0-9\-]+\.htm(?:l)?$",
    re.IGNORECASE,
)
NOTICE_HTML_ALLOW = re.compile(
    r"/grants/guide/notice-files/NOT-[A-Z0-9\-]+\.htm(?:l)?$",
    re.IGNORECASE,
)
# General Guide pages (include .cfm because index/search pages use it)
GUIDE_PAGE_ALLOW = re.compile(
    r"/grants/guide/.*\.(?:htm|html|cfm)$",
    re.IGNORECASE,
)
# Broader application pages under /grants or /grants-process
GENERIC_APP_ALLOW = re.compile(
    r"/grants(?:-process)?/.*\.(?:htm|html)$",
    re.IGNORECASE,
)

# ---------------- Helpers ----------------
def is_allowed_url(url: str) -> bool:
    try:
        u = urlparse(url)
    except Exception:
        return False
    if u.scheme.lower() not in ALLOWED_SCHEMES:
        return False
    host = (u.netloc or "").lower()
    if not any(host.endswith(d) for d in ALLOWED_DOMAINS):
        return False
    # Skip obvious non-HTML assets during crawling (ingester will discover PDFs from HTML pages)
    if u.path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".css", ".js", ".xml", ".rss", ".pdf")):
        return False
    return True

def should_enqueue(url: str) -> bool:
    if not is_allowed_url(url):
        return False
    path = urlparse(url).path
    # Explicitly include FOA & Notice pages
    if FOA_HTML_ALLOW.search(path) or NOTICE_HTML_ALLOW.search(path):
        return True
    # Include general Guide pages (to discover FOAs)
    if GUIDE_PAGE_ALLOW.search(path):
        return True
    # Include general application pages (some link to forms/updates)
    if GENERIC_APP_ALLOW.search(path):
        return True
    return False

def extract_links(base_url: str, html: str):
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        links.append(urljoin(base_url, a["href"]))
    return links

def fetch_html(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()
    # Treat as HTML if server says so OR if URL ends with common HTML-ish extensions
    if "text/html" in ctype or url.lower().endswith((".htm", ".html", ".cfm")):
        return r.text
    return ""

# ---------------- Main crawl ----------------
def crawl_and_ingest():
    visited_ingest = set()  # for ingest_url recursion (avoid repeated PDFs, etc.)
    enqueued = set()
    q = queue.Queue()

    for s in START_URLS:
        q.put((s, 0))
        enqueued.add(s)

    pages_processed = 0
    while not q.empty() and pages_processed < MAX_PAGES_PER_RUN:
        url, depth = q.get()

        # 1) Fetch HTML for crawling
        try:
            html = fetch_html(url)
        except Exception as e:
            print(f"❌ Fetch error: {url} — {e}")
            continue

        # 2) Extract and enqueue links
        if html:
            for link in extract_links(url, html):
                if link in enqueued:
                    continue
                if depth + 1 <= MAX_DEPTH and should_enqueue(link):
                    q.put((link, depth + 1))
                    enqueued.add(link)

        # 3) Ingest (handles HTML text + auto-discovers and ingests NIH PDFs)
        try:
            ingest_url(url, visited_ingest)
        except Exception as e:
            print(f"❌ Ingest error: {url} — {e}")

        pages_processed += 1
        time.sleep(CRAWL_DELAY_SEC)  # be polite

    print(f"✅ Crawl complete. Pages processed: {pages_processed}, total queued: {len(enqueued)}")

if __name__ == "__main__":
    crawl_and_ingest()
