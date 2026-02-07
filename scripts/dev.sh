#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

cd "$ROOT_DIR"

BACKEND_CMD=("$ROOT_DIR/scripts/dev-backend.sh")
FRONTEND_CMD=(npm run dev:frontend)

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

(
  cd "$ROOT_DIR"
  "${BACKEND_CMD[@]}"
) &
BACKEND_PID=$!

(
  cd "$ROOT_DIR"
  "${FRONTEND_CMD[@]}"
) &
FRONTEND_PID=$!

# Wait for frontend to be ready before opening the browser.
FRONTEND_URL="http://localhost:5173/"
for _ in {1..60}; do
  if curl -sSf "$FRONTEND_URL" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if command -v open >/dev/null 2>&1; then
  open "$FRONTEND_URL"
fi

wait
