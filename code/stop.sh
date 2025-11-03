#!/usr/bin/env bash
# Stop active profiling harness sessions along with lingering Nsight/perf workers.

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
SESSION_ROOT="$CODE_ROOT/profile_runs/harness"

found=0
TOTAL_STEPS=3
progress 1 "$TOTAL_STEPS" "Scanning PID files"

if [[ -d "$SESSION_ROOT" ]]; then
  shopt -s nullglob
  for pid_file in "$SESSION_ROOT"/*.pid; do
    [[ -f "$pid_file" ]] || continue
    pid=$(<"$pid_file")
    if [[ -z "$pid" ]]; then
      echo "Skipping empty PID file: $pid_file"
      continue
    fi
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping profiling harness process $pid (recorded in $(basename "$pid_file"))"
      kill "$pid" || true
      found=1
    else
      echo "Process $pid (from $(basename "$pid_file")) is not running"
    fi
    rm -f "$pid_file"
  done
  shopt -u nullglob
fi

progress 2 "$TOTAL_STEPS" "Terminating profiler daemons"

if command -v pkill >/dev/null 2>&1; then
  # Profile harness and Nsight helpers occasionally spawn extra workers.
  if pkill -f "profile_harness.py" >/dev/null 2>&1; then
    echo "Issued pkill for profile_harness.py"
    found=1
  fi
  if pkill -f "master_profile.py" >/dev/null 2>&1; then
    echo "Issued pkill for master_profile.py"
    found=1
  fi
  if pkill -f 'torch/_inductor/compile_worker' >/dev/null 2>&1; then
    echo "Issued pkill for torch/_inductor/compile_worker"
    found=1
  fi
  if pkill -f '/opt/nvidia/nsight-compute' >/dev/null 2>&1 || pkill -x ncu >/dev/null 2>&1; then
    echo "Issued pkill for Nsight Compute"
    found=1
  fi
  if pkill -f '/opt/nvidia/nsight-systems' >/dev/null 2>&1 || pkill -x nsys >/dev/null 2>&1; then
    echo "Issued pkill for Nsight Systems"
    found=1
  fi
  if pkill -x perf >/dev/null 2>&1; then
    echo "Issued pkill for perf"
    found=1
  fi
fi

if [[ $found -eq 0 ]]; then
  echo "No active profiling processes found."
else
  echo "Profiling processes terminated."
fi

progress 3 "$TOTAL_STEPS" "Cleanup complete"
