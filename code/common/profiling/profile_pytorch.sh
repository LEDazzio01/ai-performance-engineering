#!/bin/bash
# profile_pytorch.sh - Standard PyTorch profiling wrapper
# Usage: ./profile_pytorch.sh <python_script> [args...]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <python_script> [args...]"
    echo "Example: $0 ./training.py --batch-size 32"
    exit 1
fi

SCRIPT=$1
shift  # Remove first argument, keep the rest
ARGS="$@"

CHAPTER=$(basename $(pwd))
OUTPUT_DIR="../../results/${CHAPTER}"
mkdir -p "${OUTPUT_DIR}"

BASENAME=$(basename ${SCRIPT} .py)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "==================================================================="
echo "PyTorch Profiling: ${BASENAME}"
echo "==================================================================="

# Check if script exists
if [ ! -f "${SCRIPT}" ]; then
    echo "Error: ${SCRIPT} does not exist"
    exit 1
fi

# 1. Nsight Systems profiling with PyTorch
echo ""
echo "Step 1/2: Running Nsight Systems (CUDA + PyTorch profiling)..."
NSYS_OUTPUT="${OUTPUT_DIR}/${BASENAME}_pytorch_${TIMESTAMP}"
nsys profile \
    --output="${NSYS_OUTPUT}" \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    --python-sampling=true \
    python3 "${SCRIPT}" ${ARGS}

echo "  Timeline saved to: ${NSYS_OUTPUT}.nsys-rep"
echo "  View with: nsys-ui ${NSYS_OUTPUT}.nsys-rep"

# 2. PyTorch built-in profiler (creates Chrome trace)
echo ""
echo "Step 2/2: Running PyTorch built-in profiler..."
PYTORCH_TRACE="${OUTPUT_DIR}/${BASENAME}_trace_${TIMESTAMP}.json"
echo "  Note: Script must use torch.profiler for this to work"
echo "  Trace will be saved to: ${PYTORCH_TRACE}"

# 3. Generate summary
echo ""
echo "==================================================================="
echo "Profiling Complete!"
echo "==================================================================="
echo "Results directory: ${OUTPUT_DIR}"
echo ""
echo "To view results:"
echo "  Timeline:      nsys-ui ${NSYS_OUTPUT}.nsys-rep"
echo "  Chrome trace:  chrome://tracing (load ${PYTORCH_TRACE})"
echo ""
echo "For memory profiling, use:"
echo "  python3 -m torch.utils.bottleneck ${SCRIPT} ${ARGS}"

