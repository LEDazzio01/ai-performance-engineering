#!/bin/bash
# demo_both_examples.sh
# 
# Demonstrates BOTH pipelining examples:
# 1. Memory pipeline (CUDA) - teaches concepts
# 2. GPUDirect Storage (Python) - production GDS

set -e  # Exit on error

echo "============================================================"
echo "Chapter 10: Pipelining Demonstrations"
echo "============================================================"
echo ""
echo "This script demonstrates two complementary examples:"
echo "  1. Memory Pipeline (CUDA) - GPU memory → GPU memory"
echo "  2. GPUDirect Storage (Python) - Storage → GPU memory"
echo ""
echo "============================================================"
echo ""

# Check if CUDA example is compiled
if [ ! -f "./double_buffered_pipeline" ]; then
    echo "Building CUDA memory pipeline example..."
    make
    echo ""
fi

echo "[1/2] Memory Pipeline (CUDA) - Teaching Simulator"
echo "------------------------------------------------------------"
echo "Purpose: Demonstrate pipelining concepts"
echo "Data path: GPU Memory → GPU Memory"
echo "API: cuda::memcpy_async"
echo "Works on: Any CUDA GPU"
echo ""

./double_buffered_pipeline 512 512 512

echo ""
echo "============================================================"
echo ""

echo "[2/2] GPUDirect Storage (Python) - Real cuFile API"
echo "------------------------------------------------------------"
echo "Purpose: Production GDS implementation"
echo "Data path: NVMe Storage → GPU Memory (direct)"
echo "API: cufile.cuFileRead/Write"
echo "Works on: GDS-enabled systems (or simulation)"
echo ""

python3 cufile_gds_example.py

echo ""
echo "============================================================"
echo "Summary"
echo "============================================================"
echo ""
echo "Both examples completed successfully!"
echo ""
echo "✓ Memory Pipeline: Demonstrated async pipelining concepts"
echo "✓ GPUDirect Storage: Showed traditional vs GDS performance"
echo ""
echo "Key Takeaway:"
echo "  - CUDA example teaches HOW pipelining works (universal)"
echo "  - Python example shows WHERE to use it (storage I/O)"
echo ""
echo "For more information:"
echo "  - Memory Pipeline: README.md (Memory Pipeline section)"
echo "  - GPUDirect Storage: CUFILE_GDS_GUIDE.md"
echo "  - Quick start: CUFILE_README.md"
echo "============================================================"

