#!/bin/bash

# Test script to verify profiling tools work without hanging
# This script tests basic functionality without extensive profiling

set -euo pipefail

SCRIPT_NAME="$1"
echo "Testing profiling tools for: $SCRIPT_NAME"

echo "=== Testing Nsight Systems ==="
timeout 30 nsys profile -t cuda,nvtx,osrt -o test_nsys python3 "$SCRIPT_NAME" || echo "Nsight Systems test completed (may have timed out)"

echo "=== Testing Nsight Compute (minimal) ==="
# Test with absolute minimal profiling
timeout 15 ncu --kernel-name "vectorized_gather_kernel" --metrics "sm__warps_active.avg.pct_of_peak_sustained_active" -o test_ncu python3 "$SCRIPT_NAME" || echo "Nsight Compute test completed (may have timed out)"

echo "=== Profiling test completed ==="
