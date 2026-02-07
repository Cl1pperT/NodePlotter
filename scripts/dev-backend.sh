#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

PYTHON_BIN="$ROOT_DIR/backend/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing venv at backend/.venv. Run: python3 -m venv backend/.venv && pip install -r backend/requirements.txt" >&2
  exit 1
fi

if command -v sysctl >/dev/null 2>&1; then
  if [[ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" == "1" ]] && command -v arch >/dev/null 2>&1; then
    exec arch -arm64 "$PYTHON_BIN" -m uvicorn app.main:app \
      --reload \
      --host 0.0.0.0 \
      --port 8000 \
      --app-dir backend \
      --reload-exclude "backend/.venv/*"
  fi
fi

exec "$PYTHON_BIN" -m uvicorn app.main:app \
  --reload \
  --host 0.0.0.0 \
  --port 8000 \
  --app-dir backend \
  --reload-exclude "backend/.venv/*"
