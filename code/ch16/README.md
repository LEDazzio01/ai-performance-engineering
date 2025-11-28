# Chapter 16 - Production Inference Optimization

## Summary
Focuses on real-world inference services: paged attention, Flash SDP, FP8 serving, telemetry hooks, schedulers, and Blackwell-friendly load-test harnesses.

## Learning Goals
- Profile large decoder workloads to spot hotspots before deploying models.
- Adopt paged attention, Flash SDP, and piecewise compilation to hit latency targets.
- Integrate FP8 quantization, symmetric memory, and cache monitoring in serving loops.
- Simulate production loads (multi-node, MoE) while validating accuracy via perplexity checks.

## Directory Layout
| Path | Description |
| --- | --- |
| `inference_optimizations_blackwell.py`, `inference_profiling.py`, `inference_server_load_test.py`, `inference_serving_8xb200.py` | Top-level orchestration scripts for profiling and load testing multi-GPU inference deployments. |
| `baseline_flash_sdp.py`, `optimized_flash_sdp.py`, `baseline_paged_attention.py`, `optimized_paged_attention.py` | Attention kernels that compare naive implementations vs Flash/paged variants. |
| `baseline_piece_graphs.py`, `optimized_piece_graphs.py`, `baseline_regional_compilation.py`, `optimized_regional_compilation.py` | Piecewise graph capture and regional compilation for stable low-latency decode. |
| `fp8_transformer_engine.py`, `test_fp8_quantization_real.py`, `symmetric_memory_inference.py`, `multi_gpu_validation.py` | Serving-time FP8 and symmetric-memory validations to guarantee accuracy and NVLink efficiency. |
| `moe_performance_benchmark.py`, `synthetic_moe_inference_benchmark.py`, `moe_workload.py` | MoE inference harnesses that stress router placement and per-expert batching. |
| `cache_monitoring.py`, `dcgm_prometheus_exporter.py`, `scheduler.py`, `perplexity_eval.py` | Telemetry, scheduling, and accuracy utilities wired into the inference pipeline. |
| `compare.py`, `requirements.txt`, `Makefile`, `expectations_gb10.json` | Harness entry and dependencies for inference-focused verification. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ch16
python compare.py --profile none
python tools/cli/aisp bench list-targets --chapter ch16
python tools/cli/aisp bench run --targets ch16 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_gb10.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python optimized_paged_attention.py --profile minimal` yields fewer page faults and improved throughput relative to the baseline script.
- `python symmetric_memory_inference.py --validate` confirms NVLink-backed KV replicas stay in sync with negligible skew.
- `python inference_server_load_test.py --duration 120` exercises the scheduler and should report stable TTFT/TPOT metrics after warm-up.

## Notes
- Little's Law capacity planning: the load-test harness emits prompt/output token percentiles, observed QPS, and a GPU estimate on every run. Supply per-phase speeds with `--prefill-tokens-per-s` and `--decode-tokens-per-s` (optionally `--tokens-per-gpu`) to size clusters quickly.
- `requirements.txt` pins Transformer Engine/Flash attention combos validated on CUDA 13; rerun `setup.sh` when upgrading toolchains.
- Telemetry helpers (`dcgm_prometheus_exporter.py`, `cache_monitoring.py`) are optional and degrade gracefully when DCGM/Prometheus are absent.
- Example:
  ```bash
  torchrun --nproc_per_node=8 ch16/inference_server_load_test.py \
    --duration 60 --target-qps 400 --prefill-tokens-per-s 42000 \
    --decode-tokens-per-s 2500 --capacity-headroom 0.35 --output-json ch16/results.json
  ```
  Output excerpt:
  ```
  Little's Law capacity plan | QPS=382.4 | Prefill=42000 tok/s | Decode=2500 tok/s | Tokens/GPU=2900 tok/s | Headroom=35%
    P50: service=118.4 ms (prefill=23.8 ms, decode=94.6 ms) → GPUs=1.56, GPUs+headroom=2.11
    P95: service=214.9 ms (prefill=40.3 ms, decode=174.6 ms) → GPUs=2.84, GPUs+headroom=3.83
  ```
- The aggregated JSON now contains a `capacity_plan` block so you can archive the exact inputs that produced the estimate.
- All token/sec inputs (`--prefill-tokens-per-s`, `--decode-tokens-per-s`, `--tokens-per-gpu`) must be *per GPU* sustained numbers. If you have only cluster-wide throughput, divide by GPU count before feeding it to the planner; otherwise the Little’s Law math will undercount needed GPUs.

## Standalone Capacity CLI
- Use `python ch16/capacity_planner.py --results ch16/results.json` to re-run the estimation offline or share the JSON with the ops team. Override any metric inline (for example, `--qps 500 --tokens-per-gpu 3500`).
- You can also skip the results file entirely by passing raw metrics: `python ch16/capacity_planner.py --qps 55 --tokens-per-gpu 3200 --prompt-p50 900 --prompt-p95 1800 --gen-p50 220 --gen-p95 400 --prefill-tokens-per-s 40000 --decode-tokens-per-s 2000`.
- Add `--installed-gpus 64` to highlight when headroom-adjusted requirements exceed today's fleet.

## Notes
- `dcgm_prometheus_exporter.py` emits per-GPU metrics consumable by Prometheus/Grafana without extra setup.
- `cache_monitoring.py` can be run standalone to sanity-check allocator health between runs.
