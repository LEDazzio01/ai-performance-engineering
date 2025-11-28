#!/usr/bin/env bash
set -euo pipefail

# Stop the dashboard frontend/backend (and any stray processes).
# Usage: ./stop-dashboard.sh
# Env:
#   PORT (frontend dev server port, default 6969)
#   BACKEND_PORT (Python backend port, default 6970)

PORT="${PORT:-6969}"
BACKEND_PORT="${BACKEND_PORT:-6970}"

PORTS=("$PORT" "$BACKEND_PORT")
PATTERNS=(
  "python -m tools.dashboard.server"
  "tools/dashboard/server.py"
  "tools/dashboard-next"
)

declare -A SEEN
PIDS=()

add_pid() {
  local pid="$1" reason="$2"
  [[ "$pid" =~ ^[0-9]+$ ]] || return
  if [[ -z "${SEEN[$pid]:-}" ]]; then
    SEEN["$pid"]="$reason"
    PIDS+=("$pid")
  elif [[ "${SEEN[$pid]}" != *"$reason"* ]]; then
    SEEN["$pid"]+=", $reason"
  fi
}

collect_port_pids() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    while IFS= read -r pid; do add_pid "$pid" "port:$port"; done < <(lsof -t -i :"$port" 2>/dev/null || true)
  fi
  if command -v fuser >/dev/null 2>&1; then
    local fuser_output
    fuser_output="$(fuser -n tcp "$port" 2>/dev/null || true)"
    if [[ -n "$fuser_output" ]]; then
      for pid in $fuser_output; do add_pid "$pid" "port:$port"; done
    fi
  fi
}

collect_pattern_pids() {
  local pattern="$1"
  if command -v pgrep >/dev/null 2>&1; then
    while IFS= read -r pid; do add_pid "$pid" "pattern:$pattern"; done < <(pgrep -f -- "$pattern" || true)
  else
    while IFS= read -r line; do
      local pid="${line%% *}"
      local cmd="${line#* }"
      [[ "$cmd" == *"$pattern"* ]] && add_pid "$pid" "pattern:$pattern"
    done < <(ps -eo pid=,args=)
  fi
}

echo "[stop] Finding dashboard processes..."
for port in "${PORTS[@]}"; do
  collect_port_pids "$port"
done
for pattern in "${PATTERNS[@]}"; do
  collect_pattern_pids "$pattern"
done

if [[ ${#PIDS[@]} -eq 0 ]]; then
  echo "[stop] No dashboard processes found."
  exit 0
fi

echo "[stop] Terminating ${#PIDS[@]} process(es):"
for pid in "${PIDS[@]}"; do
  echo "  - $pid (${SEEN[$pid]})"
done

kill -TERM "${PIDS[@]}" 2>/dev/null || true
sleep 1

STILL=()
for pid in "${PIDS[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    STILL+=("$pid")
  fi
done

if [[ ${#STILL[@]} -gt 0 ]]; then
  echo "[stop] Forcing ${#STILL[@]} remaining process(es): ${STILL[*]}"
  kill -9 "${STILL[@]}" 2>/dev/null || true
else
  echo "[stop] All dashboard processes stopped."
fi
