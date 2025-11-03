# Tools Quick Reference

A quick reference for all the new tools created to address the gaps in KNOWN_GAPS.md.

## 🔍 Memory Profiling

### Profile a workload
```bash
python3 tools/memory_profiler.py <script.py> \
    --trace memory_trace.json \
    --sort self_cuda_memory_usage
```

### Analyze memory profiles
```bash
python3 tools/memory_analyzer.py \
    --input memory_profiles/ \
    --output memory_summary.md
```

---

## ✅ Accuracy Testing

### Create evaluation datasets
```bash
python3 tools/create_eval_datasets.py \
    --output-dir eval_datasets \
    --vocab-size 50000 \
    --num-tokens 50000
```

### Compare precision modes
```bash
python3 tools/compare_precision_accuracy.py \
    --dataset eval_datasets/structured_tokens.txt \
    --precisions fp16 bf16 fp8 \
    --output comparison.md
```

### Run full accuracy test suite
```bash
./tools/run_accuracy_tests.sh [output_dir]
```

---

## 🔬 Nsight Profiling

### Profile all workloads
```bash
./tools/profile_all_workloads.sh [output_dir]
```

### Profile specific workloads
```bash
python3 tools/nsys_profile_workload.py \
    --output-dir nsys_profiles \
    --workloads fp8_benchmark flex_attention
```

### List available workloads
```bash
python3 tools/nsys_profile_workload.py --list
```

---

## ⚡ Power & Cost Analysis

### Monitor power during execution
```bash
python3 tools/power_monitor.py \
    --output-json power_metrics.json \
    -- python3 my_benchmark.py
```

### Calculate cost per token
```bash
python3 tools/calculate_cost_per_token.py \
    --power-json power_metrics.json \
    --throughput 5000 \
    --electricity-cost 0.16 \
    --output cost_report.md
```

### Run benchmark with integrated power monitoring
```bash
./tools/benchmark_with_power.sh \
    ch16/test_gpt_large_optimized.py \
    "--batch-size 1 --seq-len 2048" \
    output_dir
```

---

## 📊 Regression Detection

### Detect regressions
```bash
python3 tools/detect_regressions.py \
    --artifact-dir benchmark_runs \
    --output regression_report.md \
    --fail-on-regression
```

### Run CI benchmark check
```bash
./tools/ci_benchmark_check.sh \
    config.json \
    benchmark_runs \
    commit_hash
```

---

## 🚀 8x B200 Load Testing

### Run load test
```bash
./tools/orchestrate_8xb200_load_test.sh \
    <duration_sec> \
    <target_qps> \
    [output_dir]
```

### Example: 5-minute test at 100 QPS
```bash
./tools/orchestrate_8xb200_load_test.sh 300 100
```

### Capture NVLink bandwidth during stress
```bash
./tools/capture_nvlink_during_inference.sh \
    <duration_sec> \
    <output_dir> \
    <workload_type>  # inference, training, or bandwidth

# Example: 60s inference stress with NVLink monitoring
./tools/capture_nvlink_during_inference.sh 60 nvlink_results inference
```

### Profile 40B model with Nsight Systems
```bash
./tools/profile_40b_8gpu_nsight.sh \
    <model_size> \
    <duration_sec> \
    <output_dir>

# Example: Profile 40B model for 30 seconds
./tools/profile_40b_8gpu_nsight.sh 40B 30 profile_output

# View in Nsight UI
nsys-ui profile_output/profile_40B_8gpu.nsys-rep
```

### Run comprehensive 8-GPU benchmark suite
```bash
./run_comprehensive_8gpu_benchmark.sh

# Results: 8gpu_benchmark_results_YYYYMMDD_HHMMSS/
# Archive: 8gpu_benchmark_results_YYYYMMDD_HHMMSS.tar.gz
```

### See full documentation
```bash
cat docs/8xb200_load_testing_guide.md
```

---

## 🔄 Complete Test Suite

### Run all tests (includes memory profiling)
```bash
./run_all_tests.sh
```

Results will be in `test_results_YYYYMMDD_HHMMSS/` including:
- Test results
- Memory profiles
- Performance analysis

---

## 📦 Common Workflows

### Workflow 1: Profile and Optimize

```bash
# 1. Profile with Nsight
./tools/profile_all_workloads.sh profiles_baseline

# 2. Make optimizations
# ... edit code ...

# 3. Profile again
./tools/profile_all_workloads.sh profiles_optimized

# 4. Compare results
# Review summary files in both directories
```

### Workflow 2: Accuracy Validation

```bash
# 1. Create datasets
python3 tools/create_eval_datasets.py

# 2. Run comparison
python3 tools/compare_precision_accuracy.py \
    --dataset eval_datasets/structured_tokens.txt \
    --precisions fp16 fp8

# 3. Review report
cat precision_comparison.md
```

### Workflow 3: Cost Analysis

```bash
# 1. Run benchmark with power monitoring
./tools/benchmark_with_power.sh \
    ch16/test_gpt_large_optimized.py

# 2. Review cost analysis
cat power_benchmark_*/cost_analysis.md
```

### Workflow 4: CI Integration

```bash
# Run in CI pipeline
./tools/ci_benchmark_check.sh \
    docs/examples/continuous_benchmark.json \
    benchmark_runs \
    $CI_COMMIT_SHA

# Fails with exit code 1 if regressions detected
```

### Workflow 5: Production Load Test

```bash
# 1. Run comprehensive load test
./tools/orchestrate_8xb200_load_test.sh 600 200 prod_test

# 2. Review summary
cat prod_test/SUMMARY.md

# 3. Check cost analysis
cat prod_test/cost_analysis.md

# 4. Archive results
tar czf prod_test_$(date +%Y%m%d).tar.gz prod_test/
```

---

## 🆘 Help & Documentation

All tools support `--help`:

```bash
python3 tools/<tool_name>.py --help
./tools/<script_name>.sh --help  # if implemented
```

Comprehensive guides:
- `docs/8xb200_load_testing_guide.md` - Load testing
- `docs/moe_deployment_playbook.md` - MoE deployment, routing, autoscaling
- `docs/power_efficiency_baselines.md` - Power/cost metrics, tokens/Joule
- `docs/architecture_guides.md` - Architecture-specific tuning
- `GAPS_ADDRESSED_SUMMARY.md` - What was built
- `KNOWN_GAPS.md` - Gap status tracking

---

## 📝 Output Formats

### JSON Outputs
Most tools support `--output-json` for programmatic access:
- Power metrics
- Cost calculations
- Regression analysis
- Benchmark results

### Markdown Reports
Human-readable reports with analysis and recommendations:
- Cost analysis
- Regression reports
- Accuracy comparisons
- Load test summaries

### Chrome Traces
For detailed profiling:
- Memory profiler generates `.json` traces
- Nsight profiler generates `.nsys-rep` files
- Open in `chrome://tracing` or `nsys-ui`

---

## 🎯 Quick Decision Guide

**Need to...**

- ❓ Check memory usage → `tools/memory_profiler.py`
- ❓ Profile kernels → `tools/profile_all_workloads.sh`
- ❓ Test accuracy → `tools/run_accuracy_tests.sh`
- ❓ Calculate costs → `tools/benchmark_with_power.sh`
- ❓ Detect regressions → `tools/detect_regressions.py`
- ❓ Load test 8x B200 → `tools/orchestrate_8xb200_load_test.sh`
- ❓ Run everything → `./run_all_tests.sh`

---

## 🔧 Installation Requirements

Most tools are ready to use, but some require:

```bash
# For power monitoring
pip install nvidia-ml-py3

# For profiling (if not installed)
# Download from: https://developer.nvidia.com/nsight-systems
```

---

## ✅ All Tools Ready

All tools have been created, tested for syntax, and documented. They are ready for use when you have access to the appropriate hardware.


