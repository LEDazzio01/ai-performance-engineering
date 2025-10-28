#!/bin/bash
# Comprehensive 8x B200 GPU Benchmark Suite
# Fills all gaps in KNOWN_GAPS.md with real hardware results

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="8gpu_benchmark_results_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

echo "=========================================="
echo "8x B200 GPU Comprehensive Benchmark Suite"
echo "Start Time: $(date)"
echo "Results Directory: ${RESULTS_DIR}"
echo "=========================================="

# Capture hardware configuration
echo "==> Capturing hardware configuration..."
nvidia-smi --query-gpu=index,name,memory.total,compute_cap,pcie.link.width.current --format=csv > "${RESULTS_DIR}/gpu_config.csv"
nvidia-smi topo -m > "${RESULTS_DIR}/nvlink_topology.txt"

# Test 1: Multi-GPU Validation
echo ""
echo "==> Test 1: Multi-GPU Tensor Parallel Validation"
cd /home/ubuntu/dev/ai-performance-engineering/code
python ch16/multi_gpu_validation.py \
    --world-size 8 \
    --model-size 8B \
    2>&1 | tee "${RESULTS_DIR}/test1_multi_gpu_validation.log"

# Test 2: Inference Server Load Test with Power Monitoring
echo ""
echo "==> Test 2: Inference Server Load Test (with power monitoring)"
# Start power monitor in background
python tools/power_monitor.py \
    --interval 0.1 \
    --output "${RESULTS_DIR}/power_metrics_inference.json" \
    --command "torchrun --nproc_per_node=8 ch16/inference_server_load_test.py --world-size 8 --num-requests 1000 --batch-size 8 --seq-len 2048 --max-new-tokens 512 --output ${RESULTS_DIR}/inference_load_test.json" \
    2>&1 | tee "${RESULTS_DIR}/test2_inference_load_power.log"

# Test 3: NVLink Bandwidth During Inference
echo ""
echo "==> Test 3: NVLink Bandwidth Benchmark (8 GPUs)"
python ch4/bandwidth_benchmark_suite_8gpu.py \
    --output-json "${RESULTS_DIR}/nvlink_bandwidth_8gpu.json" \
    2>&1 | tee "${RESULTS_DIR}/test3_nvlink_bandwidth.log"

# Test 4: Memory Profiling on Large Model
echo ""
echo "==> Test 4: Memory Profiling - Large Model (40B)"
python tools/memory_profiler.py \
    --output "${RESULTS_DIR}/memory_profile_40b.json" \
    --chrome-trace "${RESULTS_DIR}/memory_trace_40b.json" \
    python ch16/test_gpt_large_optimized.py \
    --model-size 40B \
    --batch-size 4 \
    --seq-len 4096 \
    --fp8-mode auto \
    --attention-backend flex \
    --output "${RESULTS_DIR}/gpt_40b_profiled.json" \
    2>&1 | tee "${RESULTS_DIR}/test4_memory_profile.log"

# Test 5: FP32 vs FP8 Accuracy Comparison
echo ""
echo "==> Test 5: Perplexity Evaluation - FP32 vs FP8"
# FP32 baseline
echo "  Running FP32 baseline..."
python ch16/perplexity_eval.py \
    --model-size 8B \
    --precision fp32 \
    --output "${RESULTS_DIR}/perplexity_fp32.json" \
    2>&1 | tee "${RESULTS_DIR}/test5_perplexity_fp32.log"

# FP16 comparison
echo "  Running FP16..."
python ch16/perplexity_eval.py \
    --model-size 8B \
    --precision fp16 \
    --output "${RESULTS_DIR}/perplexity_fp16.json" \
    2>&1 | tee "${RESULTS_DIR}/test5_perplexity_fp16.log"

# FP8 with transformer_engine
echo "  Running FP8 (transformer_engine)..."
python ch16/perplexity_eval.py \
    --model-size 8B \
    --precision fp8 \
    --output "${RESULTS_DIR}/perplexity_fp8.json" \
    2>&1 | tee "${RESULTS_DIR}/test5_perplexity_fp8.log"

# Test 6: MoE Performance Benchmark with Power
echo ""
echo "==> Test 6: MoE Performance Benchmark (with power monitoring)"
python tools/power_monitor.py \
    --interval 0.1 \
    --output "${RESULTS_DIR}/power_metrics_moe.json" \
    --command "python ch16/moe_performance_benchmark.py --output ${RESULTS_DIR}/moe_benchmark.json" \
    2>&1 | tee "${RESULTS_DIR}/test6_moe_power.log"

# Test 7: Large Model Multi-GPU Test (40B on 8 GPUs)
echo ""
echo "==> Test 7: Large Model Inference (40B, 8-GPU Tensor Parallel)"
torchrun --nproc_per_node=8 ch16/test_gpt_large_optimized.py \
    --model-size 40B \
    --batch-size 8 \
    --seq-len 8192 \
    --fp8-mode transformer-engine \
    --attention-backend flex \
    --tensor-parallel 8 \
    --output "${RESULTS_DIR}/gpt_40b_8gpu_tp.json" \
    2>&1 | tee "${RESULTS_DIR}/test7_40b_8gpu.log"

# Test 8: Inference Server with Multiple Workload Sizes
echo ""
echo "==> Test 8: Inference Server - Multiple Workload Configurations"
for BS in 1 4 8 16; do
    for SEQ in 512 2048 8192; do
        echo "  Testing batch_size=${BS}, seq_len=${SEQ}..."
        torchrun --nproc_per_node=8 ch16/inference_server_load_test.py \
            --world-size 8 \
            --num-requests 100 \
            --batch-size ${BS} \
            --seq-len ${SEQ} \
            --max-new-tokens 256 \
            --output "${RESULTS_DIR}/inference_bs${BS}_seq${SEQ}.json" \
            2>&1 | tee "${RESULTS_DIR}/test8_inference_bs${BS}_seq${SEQ}.log"
    done
done

# Test 9: Power Efficiency Baselines (tokens/joule)
echo ""
echo "==> Test 9: Power Efficiency Analysis"
python tools/power_monitor.py \
    --interval 0.05 \
    --output "${RESULTS_DIR}/power_efficiency_8b.json" \
    --command "python ch16/test_gpt_large_optimized.py --model-size 8B --batch-size 16 --seq-len 4096 --iterations 100 --output ${RESULTS_DIR}/throughput_8b.json" \
    2>&1 | tee "${RESULTS_DIR}/test9_power_efficiency.log"

# Test 10: Continuous Benchmark Suite
echo ""
echo "==> Test 10: Running Full Continuous Benchmark Suite"
python tools/continuous_benchmark.py \
    --config docs/examples/continuous_benchmark.json \
    --output-dir "${RESULTS_DIR}/continuous_benchmarks" \
    2>&1 | tee "${RESULTS_DIR}/test10_continuous_benchmark.log"

# Generate comprehensive report
echo ""
echo "==> Generating Comprehensive Report"
python tools/report_generator.py \
    --results-dir "${RESULTS_DIR}" \
    --output "${RESULTS_DIR}/COMPREHENSIVE_REPORT.md" \
    2>&1 | tee "${RESULTS_DIR}/report_generation.log"

# Calculate power efficiency metrics
echo ""
echo "==> Calculating Power Efficiency Metrics"
python tools/analyze_results.py \
    --power-file "${RESULTS_DIR}/power_metrics_inference.json" \
    --throughput-file "${RESULTS_DIR}/inference_load_test.json" \
    --output "${RESULTS_DIR}/power_efficiency_summary.json" \
    2>&1 | tee "${RESULTS_DIR}/power_analysis.log"

# Archive results
echo ""
echo "==> Archiving results..."
tar -czf "${RESULTS_DIR}.tar.gz" "${RESULTS_DIR}"
echo "Results archived to: ${RESULTS_DIR}.tar.gz"

# Summary
echo ""
echo "=========================================="
echo "Benchmark Suite Complete!"
echo "End Time: $(date)"
echo "Results: ${RESULTS_DIR}"
echo "Archive: ${RESULTS_DIR}.tar.gz"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review comprehensive report: ${RESULTS_DIR}/COMPREHENSIVE_REPORT.md"
echo "2. Update KNOWN_GAPS.md with results"
echo "3. Commit results to git"

exit 0

