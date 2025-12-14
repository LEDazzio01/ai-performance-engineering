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
| `compare.py`, `requirements.txt`, `Makefile`, `expectations_b200.json` | Harness entry and dependencies for inference-focused verification. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python ch16/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch16
python -m cli.aisp bench run --targets ch16 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_b200.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python optimized_paged_attention.py --profile minimal` yields fewer page faults and improved throughput relative to the baseline script.
- `python symmetric_memory_inference.py --validate` confirms NVLink-backed KV replicas stay in sync with negligible skew.
- `python inference_server_load_test.py --duration 120` exercises the scheduler and should report stable TTFT/TPOT metrics after warm-up.

## Notes
- `dcgm_prometheus_exporter.py` emits per-GPU metrics consumable by Prometheus/Grafana without extra setup.
- `cache_monitoring.py` can be run standalone to sanity-check allocator health between runs.
