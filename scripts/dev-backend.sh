#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

PYTHON_BIN="$ROOT_DIR/backend/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing venv at backend/.venv. Run: python3 -m venv backend/.venv && pip install -r backend/requirements.txt" >&2
  exit 1
fi

DEFAULT_PORT=8000
PORT="${BACKEND_PORT:-$DEFAULT_PORT}"

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

if ! is_port_free "$PORT"; then
  if [[ -n "${BACKEND_PORT:-}" ]]; then
    echo "Requested BACKEND_PORT $PORT is already in use." >&2
    exit 1
  fi
  PORT="$(find_free_port)"
  echo "Port $DEFAULT_PORT is in use; starting backend on port $PORT."
fi

if command -v sysctl >/dev/null 2>&1; then
  if [[ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" == "1" ]] && command -v arch >/dev/null 2>&1; then
    exec arch -arm64 "$PYTHON_BIN" -m uvicorn app.main:app \
      --reload \
      --host 0.0.0.0 \
      --port "$PORT" \
      --app-dir backend \
      --reload-exclude "backend/.venv/*"
  fi
fi

exec "$PYTHON_BIN" -m uvicorn app.main:app \
  --reload \
  --host 0.0.0.0 \
  --port "$PORT" \
  --app-dir backend \
  --reload-exclude "backend/.venv/*"
