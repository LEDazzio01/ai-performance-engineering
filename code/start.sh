#!/usr/bin/env bash
# Launch the profiling harness across all registered chapters using the
# chapter-specific metric overrides defined in metrics_config.py.
#
# The harness runs in the background under nohup so long-running Nsight
# sessions survive the terminal. Logs and PID files live under
# ./profile_runs/harness. Pass additional harness arguments to tailor a run.

set -euo pipefail

progress() {
  local current=$1
  local total=$2
  local message=$3
  local width=40
  local filled=$(( current * width / total ))
  local empty=$(( width - filled ))
  printf '\r[%s%s] %s' \
    "$(printf '%*s' "$filled" '' | tr ' ' '#')" \
    "$(printf '%*s' "$empty" '' | tr ' ' '-')" \
    "$message"
  if (( current == total )); then
    printf '\n'
  fi
}

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS="$CODE_ROOT/scripts/harness/profile_harness.py"
SESSION_ROOT="$CODE_ROOT/profile_runs/harness"
mkdir -p "$SESSION_ROOT"

TOTAL_STEPS=6

PYTHON_BIN=${PYTHON:-}
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python3)
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python)
  else
    echo "ERROR: No Python interpreter found. Set \$PYTHON or install python3." >&2
    exit 1
  fi
fi

if [[ ! -f "$HARNESS" ]]; then
  echo "ERROR: Unable to locate profiling harness at $HARNESS" >&2
  exit 1
fi

progress 1 "$TOTAL_STEPS" "Using interpreter: $PYTHON_BIN"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$SESSION_ROOT/harness_${TIMESTAMP}.log"
PID_FILE="$SESSION_ROOT/harness_${TIMESTAMP}.pid"
LATEST_LOG="$SESSION_ROOT/latest.log"
LATEST_PID="$SESSION_ROOT/latest.pid"

CMD=(env PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$HARNESS")
if [[ $# -gt 0 ]]; then
  CMD+=("$@")
else
  CMD+=("--profile" "all")
fi

{
  echo "=== Profiling harness launched at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "Code Root  : $CODE_ROOT"
  echo "Interpreter: $PYTHON_BIN"
  echo "Command    : ${CMD[*]}"
  echo "Log file   : $LOG_FILE"
  echo "PID file   : $PID_FILE"
} > "$LOG_FILE"

ln -sf "$(basename "$LOG_FILE")" "$LATEST_LOG"
progress 2 "$TOTAL_STEPS" "Logging to: $LOG_FILE"

cd "$CODE_ROOT"
progress 3 "$TOTAL_STEPS" "Launching profiling harness"

nohup "${CMD[@]}" >> "$LOG_FILE" 2>&1 &
HARNESS_PID=$!
echo "$HARNESS_PID" > "$PID_FILE"
ln -sf "$(basename "$PID_FILE")" "$LATEST_PID"

progress 4 "$TOTAL_STEPS" "Harness running (pid $HARNESS_PID)"

cat <<EOF

Profiling harness is running in the background.
  PID : $HARNESS_PID
  Log : $LOG_FILE

View live progress with:
  tail -f "$LATEST_LOG"

Stop the harness with:
  kill $HARNESS_PID

EOF

progress 5 "$TOTAL_STEPS" "Streaming live log and summary (Ctrl+C to stop)"
echo "--- Live harness log (press Ctrl+C to stop tailing; harness keeps running) ---"

tail --pid="$HARNESS_PID" -n +1 -f "$LOG_FILE" &
TAIL_PID=$!

SUMMARY_JSON="$SESSION_ROOT/latest_summary.json"

trap 'kill "$TAIL_PID" >/dev/null 2>&1 || true; exit 0' INT TERM

while kill -0 "$HARNESS_PID" >/dev/null 2>&1; do
  sleep 5
  if [[ -f "$SUMMARY_JSON" ]]; then
    success=$(jq '[.[] | select(.exit_code == 0 or .skipped == true)] | length' "$SUMMARY_JSON" 2>/dev/null)
    total=$(jq 'length' "$SUMMARY_JSON" 2>/dev/null)
    if [[ -n "$total" && "$total" -gt 0 ]]; then
      printf '\rProgress: %d/%d tasks succeeded or skipped' "$success" "$total"
    fi
  fi
done

kill "$TAIL_PID" >/dev/null 2>&1 || true

echo
echo "Harness process $HARNESS_PID exited. Check log for details."

progress 6 "$TOTAL_STEPS" "Harness completed"
