#!/usr/bin/env bash
set -euo pipefail

# Start the Next.js dashboard (and the Python backend if not already running).
# Usage: ./start-dashboard.sh
# Env:
#   PORT (frontend) default 6969
#   BACKEND_PORT default 6970
#   BACKEND_HOST default localhost
#   SKIP_BACKEND=1 to skip starting backend
#   LOG_FILE (default /tmp/nextjs-dashboard.log)
#   BACKEND_LOG_FILE (default /tmp/dashboard-backend.log)
#   CLEAN_CACHE (default 1), RUN_BUILD (default 0), TAIL_LOG (default 1)

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$ROOT_DIR/tools/dashboard-next"
LOG_FILE="${LOG_FILE:-/tmp/nextjs-dashboard.log}"
BACKEND_LOG_FILE="${BACKEND_LOG_FILE:-/tmp/dashboard-backend.log}"
CLEAN_CACHE="${CLEAN_CACHE:-1}"   # default: wipe .next/.turbo caches before start
RUN_BUILD="${RUN_BUILD:-0}"       # default: skip prod build for faster dev iteration
TAIL_LOG="${TAIL_LOG:-1}"         # default: tail the dev log after start
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}" # default to IPv4 to avoid IPv6 localhost issues
BACKEND_PORT="${BACKEND_PORT:-6970}"
SKIP_BACKEND="${SKIP_BACKEND:-0}" # set to 1 to skip auto-starting backend
BACKEND_PID=""
TAIL_PID=""

# Export backend host/port so Next.js picks up the same values for proxy rewrites.
export BACKEND_HOST BACKEND_PORT
export NEXT_PUBLIC_BACKEND_HOST="${BACKEND_HOST}"
export NEXT_PUBLIC_BACKEND_PORT="${BACKEND_PORT}"

# Ensure log files exist before tailing
touch "$LOG_FILE" "$BACKEND_LOG_FILE"

cleanup() {
  # Best-effort cleanup; ignore failures
  if [ -n "${TAIL_PID:-}" ]; then kill "$TAIL_PID" >/dev/null 2>&1 || true; fi
  if [ -n "${DEV_PID:-}" ]; then kill "$DEV_PID" >/dev/null 2>&1 || true; fi
  if [ -n "${BACKEND_PID:-}" ]; then kill "$BACKEND_PID" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT INT TERM

cd "$APP_DIR"

echo "[config] Backend: ${BACKEND_HOST}:${BACKEND_PORT}"

if ! command -v npm >/dev/null 2>&1; then
  echo "[error] npm not found. Please install Node.js 18+ (includes npm) and re-run." >&2
  echo "        Example (Ubuntu): sudo apt-get update && sudo apt-get install -y nodejs npm" >&2
  exit 1
fi

# Install dependencies if needed
if [ ! -d node_modules ]; then
  echo "[setup] Installing npm dependencies..."
  npm install
fi

# Optional cache wipe
if [ "$CLEAN_CACHE" -eq 1 ]; then
  echo "[cleanup] Removing .next/.turbo caches..."
  rm -rf .next .turbo node_modules/.cache >/dev/null 2>&1 || true
fi

if [ "$SKIP_BACKEND" -eq 0 ]; then
  if python - <<PY
import socket, sys
s = socket.socket()
try:
    s.connect(("127.0.0.1", int("${BACKEND_PORT}")))
    sys.exit(0)
except Exception:
    sys.exit(1)
finally:
    s.close()
PY
  then
    echo "[backend] Detected backend on :${BACKEND_PORT}"
  else
    echo "[backend] Starting Python server on :${BACKEND_PORT} ..."
    echo "[backend] Log: $BACKEND_LOG_FILE"
    (cd "$ROOT_DIR" && python -m tools.dashboard.server --port "${BACKEND_PORT}") >"$BACKEND_LOG_FILE" 2>&1 &
    BACKEND_PID=$!
    echo "[backend] PID: $BACKEND_PID"
    sleep 2
  fi
else
  echo "[backend] SKIP_BACKEND=1 so skipping backend check/start."
fi

# Run a build to surface compile errors early
if [ "$RUN_BUILD" -eq 1 ]; then
  echo "[build] Running npm run build to verify compilation..."
  if ! npm run build; then
    echo "[build] Build failed. See logs above or $LOG_FILE" >&2
    exit 1
  fi
fi

# Frontend port (avoid clashing with backend); default 3000
PORT="${PORT:-6969}"
echo "[frontend] Starting Next.js dev server on :$PORT ..."
echo "[frontend] Logs: $LOG_FILE"

export HOST=0.0.0.0
export PORT

if lsof -i :"$PORT" >/dev/null 2>&1; then
  echo "[warn] Port $PORT is already in use. If a previous dev server is running, stop it or set PORT=xxxx when invoking this script."
fi

# Run dev server in background, piping to log
npm run dev -- --hostname "$HOST" --port "$PORT" >>"$LOG_FILE" 2>&1 &
DEV_PID=$!
echo "[frontend] Dev server PID: $DEV_PID"

# Tail logs in foreground if requested
if [ "$TAIL_LOG" -eq 1 ]; then
  echo "[logs] Tailing frontend log ($LOG_FILE) and backend log ($BACKEND_LOG_FILE)..."
  tail -n 0 -F "$LOG_FILE" "$BACKEND_LOG_FILE" &
  TAIL_PID=$!

  wait "$DEV_PID"

  # Ensure tail exits once the dev server stops
  kill "$TAIL_PID" >/dev/null 2>&1 || true
  wait "$TAIL_PID" || true
else
  wait "$DEV_PID"
fi
