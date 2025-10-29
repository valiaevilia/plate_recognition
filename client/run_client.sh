#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python -m venv .venv >/dev/null 2>&1 || true
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

if [[ ! -f .env && -f .env.example ]]; then
  cp .env.example .env
  echo "[i] Created client/.env from example. Fill PROXY_URL & PROXY_CLIENT_TOKEN."
fi

python ./plate_model.py --video "./videos/test.mp4" --out "./out.csv" --sample-fps 5 --dedupe-window 2.0
