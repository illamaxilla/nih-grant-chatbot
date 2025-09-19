#!/bin/zsh
set -euo pipefail
cd ~/Projects/nih-grant-chatbot
source .venv/bin/activate
# rotate log per run
STAMP=$(date +"%Y%m%d-%H%M%S")
python crawl_nih.py >> "logs/crawl-$STAMP.log" 2>&1
