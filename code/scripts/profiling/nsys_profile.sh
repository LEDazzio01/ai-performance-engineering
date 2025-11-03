#!/bin/bash

# Nsight Systems Profiling Script
# Targets Blackwell B200/B300 (SM100)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

BASE_TRACE_RAW="$($PYTHON_BIN -c "import sys; sys.path.insert(0, '$SCRIPT_DIR'); from metrics_config import BASE_NSYS_TRACE_MODULES; print(','.join(BASE_NSYS_TRACE_MODULES))" 2>/dev/null || true)"
BASE_TRACE="${BASE_TRACE_RAW//$'\n'/}"
if [[ -z "$BASE_TRACE" ]]; then
    BASE_TRACE="cuda,nvtx,osrt,cublas,cudnn,nvlink"
fi

BASE_EXTRA_RAW="$($PYTHON_BIN -c "import sys; sys.path.insert(0, '$SCRIPT_DIR'); from metrics_config import BASE_NSYS_EXTRA_ARGS; print(' '.join(BASE_NSYS_EXTRA_ARGS))" 2>/dev/null || true)"
BASE_EXTRA_RAW="${BASE_EXTRA_RAW//$'\n'/}"
BASE_EXTRA_OPTS=()
if [[ -n "$BASE_EXTRA_RAW" ]]; then
    read -r -a BASE_EXTRA_OPTS <<< "$BASE_EXTRA_RAW"
fi

SCRIPT_NAME="$1"
ARCH="${2:-auto}"

# Auto-detect architecture if not specified
if [ "$ARCH" = "auto" ]; then
    ARCH="sm_100"
    if command -v nvidia-smi &> /dev/null; then
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        if [[ ! "$gpu_name" =~ B200|B300 ]]; then
            echo "⚠ Non-Blackwell GPU detected; running with sm_100 profile." >&2
        fi
    else
        echo "⚠ Unable to query GPU via nvidia-smi; assuming Blackwell profile." >&2
    fi
fi

echo "Nsight Systems profiling for $SCRIPT_NAME (Architecture: $ARCH)"

TRACE_OPTS=(
    --force-overwrite=true
    -o "nsys_profile_${ARCH}_$(date +%Y%m%d_%H%M%S)"
    -t "$BASE_TRACE"
    -s cpu
    --python-sampling=true
    --python-sampling-frequency=1000
    --cudabacktrace=true
    --cudabacktrace-threshold=0
    --stats=true
)

if ((${#BASE_EXTRA_OPTS[@]})); then
    TRACE_OPTS+=("${BASE_EXTRA_OPTS[@]}")
fi

if [[ -n "${NSYS_EXTRA_OPTS:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_OPTS=($NSYS_EXTRA_OPTS)
    TRACE_OPTS+=(${EXTRA_OPTS[@]})
fi

nsys profile "${TRACE_OPTS[@]}" python "$SCRIPT_NAME"
