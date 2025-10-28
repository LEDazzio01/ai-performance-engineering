# Migrating Workloads to NVIDIA B200 / Blackwell

This guide outlines the practical steps we follow when migrating from Ampere
(A100) or Hopper (H100) to Blackwell (B200) for inference workloads.

## 1. Environment Preparation

1. **Drivers / CUDA**: Install CUDA 13.0+ and the latest R560 driver.
2. **PyTorch**: Use PyTorch 2.9 or newer to unlock FlexAttention and FP8
   primitives.
3. **Transformer Engine**: `pip install transformer-engine[pytorch]` to access
   FP8 kernels for dense and MoE models.

## 2. Baseline Capture on Source Hardware

Run the existing benchmark suite and capture:

- Perplexity / accuracy (use `ch16/perplexity_eval.py`)
- Throughput & latency (`ch16/test_gpt_large_optimized.py`)
- Memory footprint (`tools/memory_profiler.py`)
- Power envelope (`tools/power_monitor.py`)

Store results via `tools/continuous_benchmark.py` for side-by-side comparison.

## 3. Port to Blackwell

1. **Update Model Harness**: switch to the Blackwell-configured GPT harness
   with FlexAttention and FP8: `python ch16/test_gpt_large_optimized.py --fp8-mode auto`.
2. **Enable FlexAttention**: ensure `--attention-backend auto` so the benchmark
   selects native Flex kernels when available.
3. **Validate Tensor Parallelism**: `python ch16/multi_gpu_validation.py` to
   catch sharding mistakes early.

## 4. Optimize

1. **FP8 Tuning**: Move from weight-only FP8 to transformer_engine FP8. Verify
   numerics with `perplexity_eval.py`.
2. **MoE Routing**: Measure expert balance using
   `ch16/moe_performance_benchmark.py` and adjust gating thresholds.
3. **Serving Load**: Stress the 8x B200 server with
   `torchrun --nproc_per_node=8 ch16/inference_server_load_test.py` to validate
   real-world traffic patterns.

## 5. Regression Checks

Automate nightly runs using `tools/continuous_benchmark.py` and feed outputs to
version-controlled JSON artifacts. Track:

- Performance deltas vs. previous builds
- Quality metrics (perplexity)
- Power/energy per token

## 6. Rollout Checklist

- ✅ Performance meets target (throughput, latency)
- ✅ Accuracy/perplexity matches source hardware
- ✅ Power draw within datacenter budget
- ✅ Monitoring dashboards updated (nsys traces, nvml metrics)
- ✅ Fallback plan: ability to flip to FP16/BF16 if FP8 instability observed

Following this migration workflow ensures no fabricated claims—each step ties to
runbooks and scripts in this repository.
