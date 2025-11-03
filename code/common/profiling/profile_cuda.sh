#!/bin/bash
# profile_cuda.sh - Standard CUDA profiling wrapper for all chapters
# Usage: ./profile_cuda.sh <executable> [baseline|optimized]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <executable> [baseline|optimized]"
    echo "Example: $0 ./my_kernel baseline"
    exit 1
fi

EXECUTABLE=$1
VARIANT=${2:-default}
CHAPTER=$(basename $(pwd))
OUTPUT_DIR="../../results/${CHAPTER}"
mkdir -p "${OUTPUT_DIR}"

BASENAME=$(basename ${EXECUTABLE})
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "==================================================================="
echo "CUDA Profiling: ${BASENAME} (${VARIANT})"
echo "==================================================================="

# Check if executable exists and is executable
if [ ! -x "${EXECUTABLE}" ]; then
    echo "Error: ${EXECUTABLE} is not executable or does not exist"
    exit 1
fi

# 1. Nsight Systems profiling (timeline view)
echo ""
echo "Step 1/2: Running Nsight Systems (timeline profiling)..."
NSYS_OUTPUT="${OUTPUT_DIR}/${BASENAME}_${VARIANT}_${TIMESTAMP}"
nsys profile \
    --output="${NSYS_OUTPUT}" \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    "${EXECUTABLE}"

echo "  Timeline saved to: ${NSYS_OUTPUT}.nsys-rep"
echo "  View with: nsys-ui ${NSYS_OUTPUT}.nsys-rep"

# 2. Nsight Compute profiling (kernel metrics)
echo ""
echo "Step 2/2: Running Nsight Compute (kernel metrics)..."
NCU_OUTPUT="${OUTPUT_DIR}/${BASENAME}_${VARIANT}_metrics_${TIMESTAMP}"
ncu \
    --set full \
    --export="${NCU_OUTPUT}" \
    --force-overwrite \
    "${EXECUTABLE}" 2>&1 | tee "${OUTPUT_DIR}/${BASENAME}_${VARIANT}_ncu_summary.txt"

echo "  Metrics saved to: ${NCU_OUTPUT}.ncu-rep"
echo "  View with: ncu-ui ${NCU_OUTPUT}.ncu-rep"

# 3. Generate summary report
echo ""
echo "==================================================================="
echo "Profiling Complete!"
echo "==================================================================="
echo "Results directory: ${OUTPUT_DIR}"
echo ""
echo "To view results:"
echo "  Timeline:       nsys-ui ${NSYS_OUTPUT}.nsys-rep"
echo "  Kernel metrics: ncu-ui ${NCU_OUTPUT}.ncu-rep"
echo ""
echo "For comparison, run with different variant:"
echo "  $0 ${EXECUTABLE} baseline"
echo "  $0 ${EXECUTABLE} optimized"

