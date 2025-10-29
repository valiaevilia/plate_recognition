#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python -m venv .venv >/dev/null 2>&1 || true
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

if [[ ! -f .env && -f .env.example ]]; then
  cp .env.example .env
  echo "[i] Created proxy/.env. Put NVIDIA_API_KEY there."
fi

# uvicorn will import app.py; load_dotenv will read .env
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
uvicorn app:app --host "$HOST" --port "$PORT" --reload
