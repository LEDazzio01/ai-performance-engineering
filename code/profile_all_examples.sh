#!/bin/bash
################################################################################
# Comprehensive FP4/FP6/FP8 Profiling Script
################################################################################
# 
# This script profiles all FP4/FP6/FP8 quantization examples with:
# - PyTorch Profiler (detailed kernel analysis)
# - Nsight Systems (timeline profiling)
# - Nsight Compute (kernel metrics)
#
# Usage:
#   ./profile_all_examples.sh [quick|full]
#
# Modes:
#   quick - Fast profiling (fewer iterations, limited metrics)
#   full  - Comprehensive profiling (more iterations, full metrics)
#
################################################################################

set -e  # Exit on error

# Configuration
MODE="${1:-quick}"
BASE_DIR="/home/cfregly/ai-performance-engineering/code"
OUTPUT_DIR="${BASE_DIR}/validation_results"
PROFILER_DIR="${BASE_DIR}/profiler_output"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================================
Comprehensive FP4/FP6/FP8 Profiling Script
================================================================================${NC}"

echo "Mode: $MODE"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if running on GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. This script requires NVIDIA GPU.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ GPU detected:${NC}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROFILER_DIR"

# Set iterations based on mode
if [ "$MODE" == "full" ]; then
    WARMUP_ITERS=20
    BENCH_ITERS=50
    NCU_KERNEL_COUNT=10
    echo -e "${YELLOW}Running in FULL mode (comprehensive profiling)${NC}"
else
    WARMUP_ITERS=10
    BENCH_ITERS=20
    NCU_KERNEL_COUNT=5
    echo -e "${YELLOW}Running in QUICK mode (fast profiling)${NC}"
fi

cd "$BASE_DIR"

################################################################################
# 1. FP8 Matmul Validation Script
################################################################################

echo -e "\n${BLUE}============================================================
1. FP8 Matmul (Validation Framework)
============================================================${NC}"

# PyTorch Profiler
echo -e "${GREEN}→ Running PyTorch Profiler...${NC}"
PYTHONPATH=$BASE_DIR python ch19/validate_quantization_performance.py \
    --example fp8_matmul \
    --profile \
    --iterations $BENCH_ITERS \
    2>&1 | tee "$OUTPUT_DIR/fp8_matmul_pytorch.log"

# Nsight Systems
echo -e "${GREEN}→ Running Nsight Systems...${NC}"
nsys profile \
    -o "$OUTPUT_DIR/fp8_matmul_validation" \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    python ch19/validate_quantization_performance.py \
    --example fp8_matmul \
    --iterations $BENCH_ITERS \
    2>&1 | tail -20

# Nsight Compute (if in full mode)
if [ "$MODE" == "full" ]; then
    echo -e "${GREEN}→ Running Nsight Compute (full metrics)...${NC}"
    ncu --set full \
        --target-processes all \
        --kernel-name-base function \
        --launch-skip 5 \
        --launch-count $NCU_KERNEL_COUNT \
        -o "$OUTPUT_DIR/fp8_matmul_validation_ncu" \
        -f \
        python ch19/validate_quantization_performance.py \
        --example fp8_matmul \
        --iterations 10 \
        2>&1 | tail -20
fi

################################################################################
# 2. FP8 Compiled Matmul (Direct)
################################################################################

echo -e "\n${BLUE}============================================================
2. FP8 Compiled Matmul (Direct Example)
============================================================${NC}"

# Nsight Systems
echo -e "${GREEN}→ Running Nsight Systems...${NC}"
cd ch19
PYTHONPATH=$BASE_DIR nsys profile \
    -o "$OUTPUT_DIR/fp8_compiled_matmul" \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    python fp8_compiled_matmul.py \
    2>&1 | tail -30

# Nsight Compute (roofline analysis)
if [ "$MODE" == "full" ]; then
    echo -e "${GREEN}→ Running Nsight Compute (roofline)...${NC}"
    PYTHONPATH=$BASE_DIR ncu --set roofline \
        -o "$OUTPUT_DIR/fp8_compiled_roofline" \
        -f \
        python fp8_compiled_matmul.py \
        2>&1 | tail -20
fi

cd "$BASE_DIR"

################################################################################
# 3. FP8 Transformer Benchmark
################################################################################

echo -e "\n${BLUE}============================================================
3. FP8 Transformer Benchmark (Real Testing)
============================================================${NC}"

# Nsight Systems
echo -e "${GREEN}→ Running Nsight Systems...${NC}"
cd ch16
PYTHONPATH=$BASE_DIR timeout 120 nsys profile \
    -o "$OUTPUT_DIR/fp8_transformer_benchmark" \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    python test_fp8_quantization_real.py \
    2>&1 | tail -30 || echo -e "${YELLOW}(timeout expected)${NC}"

cd "$BASE_DIR"

################################################################################
# 4. Generate Summary Report
################################################################################

echo -e "\n${BLUE}============================================================
4. Generating Summary Report
============================================================${NC}"

python3 << 'PYTHON_SCRIPT'
import json
import os
from pathlib import Path

output_dir = Path("/home/cfregly/ai-performance-engineering/code/validation_results")

# Read JSON results if available
results_file = output_dir / "fp8_matmul_results.json"
if results_file.exists():
    with open(results_file) as f:
        data = json.load(f)
    
    print("\n" + "="*80)
    print("PROFILING SUMMARY")
    print("="*80)
    
    if 'results' in data and len(data['results']) > 0:
        print(f"\nBenchmark: {data['benchmark_name']}")
        print(f"Timestamp: {data.get('timestamp', 'N/A')}")
        print("\nResults:")
        print(f"{'Precision':<12} {'Time (ms)':<12} {'TFLOPS':<10} {'Memory (MB)':<12} {'Throughput (tok/s)':<20}")
        print("-" * 80)
        
        for result in data['results']:
            tflops_str = f"{result.get('tflops', 0):.2f}" if result.get('tflops') else "N/A"
            print(f"{result['precision']:<12} {result['avg_time_ms']:<12.3f} {tflops_str:<10} "
                  f"{result['memory_allocated_mb']:<12.2f} {result['throughput_tokens_per_sec']:<20.1f}")
        
        # Speedup analysis
        if len(data['results']) > 1:
            baseline = data['results'][0]
            print("\nSpeedup Analysis:")
            for result in data['results']:
                speedup = baseline['avg_time_ms'] / result['avg_time_ms'] if result['avg_time_ms'] > 0 else 0
                print(f"  {result['precision']}: {speedup:.2f}x")

# List generated files
print("\n" + "="*80)
print("GENERATED FILES")
print("="*80)

files_info = []
for file in output_dir.iterdir():
    if file.is_file():
        size_mb = file.stat().st_size / 1024 / 1024
        files_info.append((file.name, size_mb))

files_info.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'File':<50} {'Size':<10}")
print("-" * 65)
for name, size in files_info:
    if size >= 1:
        print(f"{name:<50} {size:>8.2f} MB")
    else:
        print(f"{name:<50} {size*1024:>8.2f} KB")

total_size = sum(size for _, size in files_info)
print("-" * 65)
print(f"{'Total':<50} {total_size:>8.2f} MB")

PYTHON_SCRIPT

################################################################################
# 5. Display Viewing Instructions
################################################################################

echo -e "\n${BLUE}============================================================
5. How to View Profiling Data
============================================================${NC}"

echo -e "
${GREEN}PyTorch Profiler (Chrome Traces):${NC}
  1. Open Chrome/Chromium browser
  2. Navigate to: chrome://tracing
  3. Load files from: $PROFILER_DIR/
     - FP8_Matmul_FP16_trace.json
     - FP8_Matmul_FP8_trace.json

${GREEN}Nsight Systems (Timeline):${NC}
  nsys-ui $OUTPUT_DIR/fp8_matmul_validation.nsys-rep
  nsys-ui $OUTPUT_DIR/fp8_compiled_matmul.nsys-rep
  nsys-ui $OUTPUT_DIR/fp8_transformer_benchmark.nsys-rep

${GREEN}Nsight Compute (Kernel Metrics):${NC}
  ncu-ui $OUTPUT_DIR/fp8_matmul_validation_ncu.ncu-rep
  ncu-ui $OUTPUT_DIR/fp8_compiled_roofline.ncu-rep

${GREEN}Performance Data:${NC}
  cat $OUTPUT_DIR/fp8_matmul_results.json
  cat $OUTPUT_DIR/quantization_validation_report.md
  cat $OUTPUT_DIR/README.md
"

################################################################################
# Done!
################################################################################

echo -e "${BLUE}================================================================================
✅ PROFILING COMPLETE!
================================================================================${NC}"

echo -e "
${GREEN}Summary:${NC}
  ✓ PyTorch Profiler: Chrome traces generated
  ✓ Nsight Systems: Timeline profiles generated
  ✓ Nsight Compute: Kernel metrics collected (full mode only)
  ✓ Performance data: JSON + Markdown reports

${GREEN}Output Directory:${NC}
  $OUTPUT_DIR

${GREEN}Next Steps:${NC}
  1. Review summary: cat $OUTPUT_DIR/quantization_validation_report.md
  2. View timeline: nsys-ui $OUTPUT_DIR/fp8_matmul_validation.nsys-rep
  3. Analyze kernels: ncu-ui $OUTPUT_DIR/fp8_matmul_validation_ncu.ncu-rep
  4. Read documentation: cat INDEX_NVFP4_NVFP8.md

${YELLOW}Note:${NC} Use './profile_all_examples.sh full' for comprehensive profiling
      (longer runtime, more metrics)
"


