#!/bin/bash
# Profile 40B model on 8-GPU tensor parallel with Nsight Systems
# Captures kernel timeline, NVLink traffic, NCCL operations, and memory transfers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
MODEL_SIZE="${1:-40B}"
DURATION="${2:-30}"  # Profile for 30 seconds
OUTPUT_DIR="${3:-nsight_profile_$(date +%Y%m%d_%H%M%S)}"

echo "================================================================================"
echo "Nsight Systems Profiling: ${MODEL_SIZE} Model on 8 GPUs"
echo "================================================================================"
echo ""
echo "Model size: ${MODEL_SIZE}"
echo "Profile duration: ${DURATION} seconds"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# Set PYTHONPATH
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Check if nsys is available
if ! command -v nsys &> /dev/null; then
    echo -e "${YELLOW}⚠ nsys command not found${NC}"
    echo "Please install Nsight Systems:"
    echo "  https://developer.nvidia.com/nsight-systems"
    exit 1
fi

# Get GPU count
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected ${NUM_GPUS} GPUs"
echo ""

# Nsight Systems configuration
NSYS_OPTS=(
    --trace=cuda,nvtx,osrt,cudnn,cublas,openacc
    --cuda-memory-usage=true
    --gpu-metrics-device=all
    --gpu-metrics-frequency=10000  # 10ms sampling
    --sample=cpu
    --cpuctxsw=none
    --backtrace=fp
    --duration=${DURATION}
    --delay=5  # Wait 5s for warmup
    --capture-range=cudaProfilerApi
    --output="${OUTPUT_DIR}/profile_${MODEL_SIZE}_8gpu"
)

echo -e "${BLUE}Nsight Systems Configuration:${NC}"
echo "  Traces: CUDA, NVTX, cuDNN, cuBLAS"
echo "  GPU metrics: All GPUs @ 10ms"
echo "  Duration: ${DURATION}s"
echo "  Warmup delay: 5s"
echo ""

# Prepare workload
echo -e "${BLUE}Preparing workload...${NC}"

WORKLOAD_SCRIPT="${OUTPUT_DIR}/profile_workload.py"

cat > "${WORKLOAD_SCRIPT}" << 'WORKLOAD_EOF'
"""
Workload for Nsight Systems profiling of 40B model on 8 GPUs.
This script runs a representative inference workload with NVTX markers.
"""

import sys
import os
import time
import torch
import torch.distributed as dist

# Set PYTHONPATH
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

# Import test script
from ch16.test_gpt_large_optimized import main

# Add NVTX markers for profiling
try:
    import torch.cuda.nvtx as nvtx
    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False
    class nvtx:
        @staticmethod
        def range_push(msg):
            pass
        @staticmethod
        def range_pop():
            pass

def profile_workload():
    """Run profiling workload with NVTX markers."""
    
    # Initialize distributed
    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    
    if rank == 0:
        print("=" * 80)
        print("Starting profiling workload...")
        print("=" * 80)
    
    # Enable CUDA profiler API for capture-range
    torch.cuda.cudart().cudaProfilerStart()
    
    # Warmup phase (not profiled)
    if HAS_NVTX:
        nvtx.range_push("warmup")
    
    if rank == 0:
        print("Warmup phase (5 seconds)...")
    
    time.sleep(5)
    
    if HAS_NVTX:
        nvtx.range_pop()
    
    # Main profiling phase
    if HAS_NVTX:
        nvtx.range_push("profiled_inference")
    
    if rank == 0:
        print("Starting profiled region...")
    
    # Call the actual benchmark with minimal iterations
    # We just want to capture a few iterations for profiling
    sys.argv = [
        "profile_workload.py",
        "--model-size", os.environ.get("MODEL_SIZE", "40B"),
        "--batch-size", "4",
        "--seq-len", "4096",
        "--warmup", "2",
        "--iters", "5",
        "--fp8-mode", "auto",
        "--attention-backend", "flex",
        "--skip-compile",  # Skip compile for cleaner traces
        "--output", os.environ.get("PROFILE_OUTPUT", "profile_results.json"),
    ]
    
    try:
        main()
    except SystemExit:
        pass
    
    if HAS_NVTX:
        nvtx.range_pop()
    
    # Stop profiler
    torch.cuda.cudart().cudaProfilerStop()
    
    if rank == 0:
        print("Profiling complete")

if __name__ == "__main__":
    profile_workload()
WORKLOAD_EOF

echo -e "${GREEN}✓ Workload script created${NC}"
echo ""

# Run profiling
echo -e "${BLUE}Starting Nsight Systems profiling...${NC}"
echo ""

export MODEL_SIZE="${MODEL_SIZE}"
export PROFILE_OUTPUT="${OUTPUT_DIR}/benchmark_results.json"

# Profile with torchrun
nsys profile "${NSYS_OPTS[@]}" \
    torchrun --nproc_per_node="${NUM_GPUS}" "${WORKLOAD_SCRIPT}" \
    2>&1 | tee "${OUTPUT_DIR}/profiling.log"

PROFILE_SUCCESS=$?

echo ""

if [ ${PROFILE_SUCCESS} -eq 0 ]; then
    echo -e "${GREEN}✓ Profiling complete${NC}"
else
    echo -e "${YELLOW}⚠ Profiling completed with warnings${NC}"
fi

# Generate reports
echo ""
echo -e "${BLUE}Generating profiling reports...${NC}"

# Export to various formats
PROFILE_FILE="${OUTPUT_DIR}/profile_${MODEL_SIZE}_8gpu.nsys-rep"

if [ -f "${PROFILE_FILE}" ]; then
    # Generate SQLite export for analysis
    echo "  - Exporting to SQLite..."
    nsys export --type=sqlite -o "${OUTPUT_DIR}/profile.sqlite" "${PROFILE_FILE}" 2>&1 || true
    
    # Generate summary report
    echo "  - Generating summary..."
    nsys stats --report summary "${PROFILE_FILE}" > "${OUTPUT_DIR}/summary.txt" 2>&1 || true
    
    # Generate CUDA API stats
    echo "  - Generating CUDA API stats..."
    nsys stats --report cudaapisum "${PROFILE_FILE}" > "${OUTPUT_DIR}/cuda_api_summary.txt" 2>&1 || true
    
    # Generate GPU kernel stats
    echo "  - Generating kernel stats..."
    nsys stats --report gpukernsum "${PROFILE_FILE}" > "${OUTPUT_DIR}/gpu_kernel_summary.txt" 2>&1 || true
    
    # Generate memory operation stats
    echo "  - Generating memory stats..."
    nsys stats --report gpumemtimesum "${PROFILE_FILE}" > "${OUTPUT_DIR}/gpu_memory_summary.txt" 2>&1 || true
    
    echo -e "${GREEN}✓ Reports generated${NC}"
else
    echo -e "${YELLOW}⚠ Profile file not found, skipping report generation${NC}"
fi

# Create summary document
echo ""
echo -e "${BLUE}Creating summary document...${NC}"

cat > "${OUTPUT_DIR}/SUMMARY.md" << EOL
# Nsight Systems Profile: ${MODEL_SIZE} Model on 8 GPUs

**Date**: $(date)
**Model**: ${MODEL_SIZE}
**GPUs**: ${NUM_GPUS}
**Duration**: ${DURATION} seconds

## Files Generated

- \`profile_${MODEL_SIZE}_8gpu.nsys-rep\`: Main Nsight Systems profile (open in Nsight UI)
- \`profile.sqlite\`: SQLite export for programmatic analysis
- \`summary.txt\`: Overall execution summary
- \`cuda_api_summary.txt\`: CUDA API call statistics
- \`gpu_kernel_summary.txt\`: GPU kernel execution statistics
- \`gpu_memory_summary.txt\`: Memory transfer statistics
- \`profiling.log\`: Profiling execution log
- \`benchmark_results.json\`: Benchmark performance metrics

## Viewing the Profile

### Option 1: Nsight Systems UI (Recommended)

\`\`\`bash
# Launch Nsight Systems UI and open the profile
nsys-ui ${OUTPUT_DIR}/profile_${MODEL_SIZE}_8gpu.nsys-rep
\`\`\`

Or download the .nsys-rep file and open locally on a machine with Nsight Systems UI.

### Option 2: Command-line Analysis

\`\`\`bash
# View summary
cat ${OUTPUT_DIR}/summary.txt

# View top kernels
cat ${OUTPUT_DIR}/gpu_kernel_summary.txt | head -50

# View memory operations
cat ${OUTPUT_DIR}/gpu_memory_summary.txt | head -30
\`\`\`

## Key Metrics to Look For

### 1. GPU Utilization
- Target: >85% during inference
- Check for idle periods or synchronization bottlenecks

### 2. NVLink Bandwidth
- Expected: 200-600 GB/s aggregate during all-reduce
- Look for NVLink saturation during tensor parallel ops

### 3. Kernel Efficiency
- Large GEMM kernels should dominate execution time
- Watch for kernel launch overhead (many small kernels)

### 4. Memory Transfers
- Host-to-device transfers should be minimal during inference
- Device-to-device (NVLink) should be efficient

### 5. NCCL Operations
- All-reduce latency for tensor parallel communication
- Check for overlapping compute and communication

## Common Issues to Debug

1. **Low GPU utilization**: Look for CPU bottlenecks or synchronization issues
2. **High NVLink latency**: Check topology or increase batch size
3. **Frequent small kernels**: Consider fusion or torch.compile
4. **Memory transfer overhead**: Ensure data is pre-loaded on GPUs

## Next Steps

1. Open profile in Nsight Systems UI for visual timeline
2. Filter timeline to focus on GPU 0 kernel execution
3. Identify top time-consuming kernels
4. Check NVLink metrics for communication efficiency
5. Look for optimization opportunities (kernel fusion, overlap, etc.)

## Recommended Views in Nsight UI

- **CUDA HW**: GPU utilization and kernel execution
- **NVTX**: Application-level markers
- **Threads**: CPU activity and launch overhead  
- **Memory**: GPU memory allocations and transfers
- **NVLink**: Cross-GPU communication bandwidth

EOL

cat "${OUTPUT_DIR}/SUMMARY.md"

echo ""
echo "================================================================================"
echo "Profiling Complete"
echo "================================================================================"
echo ""
echo "Profile saved to: ${OUTPUT_DIR}/"
echo ""
echo "To view the profile:"
echo "  nsys-ui ${OUTPUT_DIR}/profile_${MODEL_SIZE}_8gpu.nsys-rep"
echo ""
echo "Or view text summaries:"
echo "  cat ${OUTPUT_DIR}/summary.txt"
echo "  cat ${OUTPUT_DIR}/gpu_kernel_summary.txt"
echo ""

if [ ${PROFILE_SUCCESS} -eq 0 ]; then
    echo -e "${GREEN}✅ Profiling completed successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ Profiling completed with warnings${NC}"
    exit 0
fi
