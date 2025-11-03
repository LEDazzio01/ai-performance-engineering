#!/bin/bash

# Nsight Compute Profiling Script
# Targets Blackwell B200/B300 (SM100)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
BASE_METRICS_RAW="$($PYTHON_BIN -c "import sys; sys.path.insert(0, '$SCRIPT_DIR'); from metrics_config import BASE_NCU_METRICS; print(','.join(BASE_NCU_METRICS))" 2>/dev/null || true)"
BASE_METRICS="${BASE_METRICS_RAW//$'\n'/}"
if [[ -z "$BASE_METRICS" ]]; then
    BASE_METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,sm__sass_average_branch_targets_threads_uniform.pct,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__sass_data_bytes_mem_shared_op_ld.sum,sm__sass_data_bytes_mem_shared_op_st.sum,sm__sass_thread_inst_executed_op_fp32_pred_on.sum,sm__sass_thread_inst_executed_op_fp16_pred_on.sum,gpu__time_duration.avg"
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

echo "Nsight Compute profiling for $SCRIPT_NAME (Architecture: $ARCH)"

IFS=',' read -ra BASE_METRIC_ARRAY <<< "$BASE_METRICS"
if [[ -n "${NCU_EXTRA_METRICS:-}" ]]; then
    IFS=',' read -ra EXTRA_METRIC_ARRAY <<< "$NCU_EXTRA_METRICS"
else
    EXTRA_METRIC_ARRAY=()
fi
unset IFS

declare -A METRIC_SEEN=()
METRIC_LIST=()
for metric in "${BASE_METRIC_ARRAY[@]}" "${EXTRA_METRIC_ARRAY[@]}"; do
    metric="${metric//[[:space:]]/}"
    if [[ -n "$metric" && -z "${METRIC_SEEN[$metric]:-}" ]]; then
        METRIC_SEEN[$metric]=1
        METRIC_LIST+=("$metric")
    fi
done

METRIC_STRING=$(IFS=','; echo "${METRIC_LIST[*]}")

KERNEL_REGEX=${NCU_KERNEL_REGEX:-"regex:cutlass3x_sm100_simt_sgemm.*|vectorized_gather_kernel"}
LAUNCH_SKIP=${NCU_LAUNCH_SKIP:-0}
LAUNCH_COUNT=${NCU_LAUNCH_COUNT:-all}

KERNEL_ARGS=()
if [[ -n "$KERNEL_REGEX" && "$KERNEL_REGEX" != "all" ]]; then
    KERNEL_ARGS+=(--kernel-name "$KERNEL_REGEX")
fi

LAUNCH_ARGS=(--launch-skip "$LAUNCH_SKIP")
if [[ -n "$LAUNCH_COUNT" && "$LAUNCH_COUNT" != "all" ]]; then
    LAUNCH_ARGS+=(--launch-count "$LAUNCH_COUNT")
fi

# Collect configured metrics using Nsight Compute profiling mode
ncu \
    --set full \
    --clock-control none \
    --metrics "$METRIC_STRING" \
    "${KERNEL_ARGS[@]}" \
    "${LAUNCH_ARGS[@]}" \
    --import-source no \
    -o "ncu_profile_${ARCH}_$(date +%Y%m%d_%H%M%S)" \
    python3 "$SCRIPT_NAME"
