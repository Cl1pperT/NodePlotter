#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/backend/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

DEFAULT_BACKEND_PORT=8000
BACKEND_PORT="${BACKEND_PORT:-$DEFAULT_BACKEND_PORT}"

is_port_free() {
  "$PYTHON_BIN" - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(0.5)
try:
  sock.connect(("127.0.0.1", port))
  sys.exit(1)
except ConnectionRefusedError:
  sys.exit(0)
except OSError:
  sys.exit(1)
finally:
  sock.close()
PY
}

find_free_port() {
  "$PYTHON_BIN" - <<'PY'
import socket

sock = socket.socket()
sock.bind(("127.0.0.1", 0))
print(sock.getsockname()[1])
sock.close()
PY
}

if ! is_port_free "$BACKEND_PORT"; then
  BACKEND_PORT="$(find_free_port)"
  echo "Port $DEFAULT_BACKEND_PORT is in use; starting backend on port $BACKEND_PORT."
fi

export BACKEND_PORT
export VITE_API_BASE_URL="http://localhost:${BACKEND_PORT}"

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
