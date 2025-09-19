import sys
from ingest_basic import ingest_url
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python ingest_one.py <NIH_GUIDE_FOA_URL>")
        sys.exit(1)
    ingest_url(sys.argv[1], visited=set())
