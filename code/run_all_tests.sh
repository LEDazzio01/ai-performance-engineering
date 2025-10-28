#!/bin/bash
# Comprehensive test runner for all Blackwell B200 optimizations
# Tests all examples on B200 hardware

set -e

export NCCL_PROTO="${NCCL_PROTO:-Simple}"
export NCCL_ALGO="${NCCL_ALGO:-Tree,Ring,NVLS}"
export NCCL_NVLINK_C2C_ENABLE="${NCCL_NVLINK_C2C_ENABLE:-1}"
export NCCL_NVLINK_TCE_ENABLE="${NCCL_NVLINK_TCE_ENABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
export NCCL_NCHANNELS_PER_NET_PEER="${NCCL_NCHANNELS_PER_NET_PEER:-8}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-4}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-67108864}" # 64MB
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
export NCCL_NTHREADS="${NCCL_NTHREADS:-512}"
export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-4}"
export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-8}"
export NCCL_LL_THRESHOLD="${NCCL_LL_THRESHOLD:-0}"
export NCCL_GRAPH_REGISTER="${NCCL_GRAPH_REGISTER:-1}"

echo "================================================================================"
echo "AI Performance Engineering - Complete Test Suite"
echo "Hardware: NVIDIA B200 (SM 10.0, 180 GB HBM3e, 148 SMs)"
echo "================================================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

RESULTS_DIR="test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

test_passed=0
test_failed=0

run_test() {
    local test_name="$1"
    local test_cmd="$2"
    local output_file="$RESULTS_DIR/${test_name}.txt"
    
    echo -n "Testing $test_name... "
    
    if ( eval "$test_cmd" ) > "$output_file" 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        ((test_passed += 1))
    else
        echo -e "${RED}FAIL${NC}"
        ((test_failed += 1))
        echo "  See: $output_file"
    fi
}

echo "================================================================================"
echo "Phase 1: Building CUDA Examples"
echo "================================================================================"
echo ""

cd "${REPO_ROOT}"

for ch_dir in ch1 ch2 ch6 ch7 ch8 ch9 ch10 ch11 ch12; do
    if [ -d "$ch_dir" ] && [ -f "$ch_dir/Makefile" ]; then
        echo "Building $ch_dir..."
        (cd $ch_dir && make clean && make) || echo "  Warning: Build issues in $ch_dir"
    fi
done

echo ""
echo "================================================================================"
echo "Phase 2: Testing CUDA Kernels"
echo "================================================================================"
echo ""

run_test "ch1_batched_gemm" "cd ch1 && ./batched_gemm_example"
run_test "ch2_nvlink" "cd ch2 && ./nvlink_c2c_p2p_blackwell"
run_test "ch7_hbm3e" "cd ch7 && ./hbm3e_peak_bandwidth"
run_test "ch10_tcgen05" "cd ch10 && ./tcgen05_blackwell"
run_test "ch10_clusters" "cd ch10 && ./cluster_group_blackwell"
run_test "ch11_basic_streams" "cd ch11 && ./basic_streams"
run_test "ch11_stream_memory" "cd ch11 && ./stream_ordered_allocator"
run_test "ch12_cuda_graphs" "cd ch12 && ./cuda_graphs"

echo ""
echo "================================================================================"
echo "Phase 3: Testing PyTorch Examples"
echo "================================================================================"
echo ""

run_test "ch1_performance_basics" "cd ch1 && python3 performance_basics.py"
run_test "ch1_performance_optimized" "timeout 120 python3 ch1/performance_basics_optimized.py"
run_test "ch14_torch_compile" "cd ch14 && timeout 300 python3 torch_compiler_examples.py"
run_test "ch14_deepseek" "cd ch14 && timeout 300 python3 deepseek_innovation_l2_bypass.py"
run_test "ch16_synthetic_moe" "cd ch16 && MOE_BENCH_QUICK=1 timeout 300 python3 synthetic_moe_inference_benchmark.py"
run_test "ch18_flex_attention" "cd ch18 && timeout 300 python3 flex_attention_native.py"

echo ""
echo "================================================================================"
echo "Phase 4: Running Comprehensive Benchmarks"
echo "================================================================================"
echo ""

run_test "benchmark_peak" "timeout 600 python3 benchmark_peak.py"

echo ""
echo "================================================================================"
echo "Phase 5: Running pytest Test Suite"
echo "================================================================================"
echo ""

if [ -f "tests/test_blackwell_optimizations.py" ]; then
    run_test "pytest_correctness" "cd tests && pytest -v -m 'not slow' test_blackwell_optimizations.py"
    run_test "pytest_performance" "cd tests && pytest -v -m 'slow' test_blackwell_optimizations.py"
else
    echo -e "${YELLOW}Warning: Test suite not found${NC}"
fi

echo ""
echo "================================================================================"
echo "Quick Performance Analysis"
echo "================================================================================"
echo ""

# Run quick analysis of test results
if command -v python3 &> /dev/null; then
    if ! python3 tools/analyze_results.py --input "$RESULTS_DIR" --quick --output "$RESULTS_DIR/analysis.md" 2>/dev/null; then
        echo "Analysis skipped - metrics extraction in progress"
    fi

    if [ -f "$RESULTS_DIR/analysis.md" ]; then
        cat "$RESULTS_DIR/analysis.md"
    fi

    # Summarise any Nsight Systems reports generated during the run
    NSYS_SUMMARY_FILE="$RESULTS_DIR/nsys_summary.txt"
    if python3 tools/nsys_summary.py --glob "profiles/*.nsys-rep" \
        --kernel-regex "attn|mma|nvjet|cublas" --top-k 8 \
        --output "$NSYS_SUMMARY_FILE" >/dev/null 2>&1; then
        if [ -s "$NSYS_SUMMARY_FILE" ]; then
            echo ""
            echo "================================================================================"
            echo "Nsight Systems Kernel Summary"
            echo "================================================================================"
            cat "$NSYS_SUMMARY_FILE"
        fi
    fi
else
    echo "Python3 not available, skipping analysis"
fi

echo ""
echo "================================================================================"
echo "Test Summary"
echo "================================================================================"
echo ""
echo -e "Tests passed: ${GREEN}$test_passed${NC}"
echo -e "Tests failed: ${RED}$test_failed${NC}"
echo "Total tests:  $((test_passed + test_failed))"
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo ""

# Generate summary report
cat > "$RESULTS_DIR/SUMMARY.txt" << EOF
AI Performance Engineering - Test Summary
==========================================

Date: $(date)
Hardware: NVIDIA B200
Software: PyTorch 2.9, CUDA 13, Triton 3.5

Test Results:
- Passed: $test_passed
- Failed: $test_failed
- Total: $((test_passed + test_failed))

Success Rate: $(awk "BEGIN {printf \"%.1f\", ($test_passed/($test_passed+$test_failed))*100}")%

See individual test outputs in this directory for details.
EOF

echo "Summary report: $RESULTS_DIR/SUMMARY.txt"
echo ""

if [ $test_failed -eq 0 ]; then
    echo -e "${GREEN}================================================================================"
    echo "ALL TESTS PASSED!"
    echo -e "================================================================================${NC}"
    exit 0
else
    echo -e "${YELLOW}================================================================================"
    echo "SOME TESTS FAILED - See results directory for details"
    echo -e "================================================================================${NC}"
    exit 1
fi
