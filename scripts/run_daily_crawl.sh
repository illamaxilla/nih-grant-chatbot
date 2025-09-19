#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Projects/nih-grant-chatbot"
source .venv/bin/activate
python crawl_nih.py >> "logs/crawl-$(date +%Y%m%d-%H%M%S).log" 2>&1
