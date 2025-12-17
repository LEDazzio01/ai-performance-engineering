# Chapter 15 - Disaggregated Inference & KV Management

## Summary
Addresses large-scale inference concerns: disaggregated compute/storage, KV-cache pooling over NVLink, continuous batching, and mixture-of-experts serving patterns.

## Learning Goals
- Benchmark monolithic vs disaggregated inference paths and quantify fabric costs.
- Design KV-cache managers that gracefully span local and remote HBM pools.
- Implement continuous batching and queueing so decode throughput stays high.
- Serve MoE models efficiently by pairing routing with optimized communication.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_inference_monolithic.py`, `optimized_inference_monolithic.py`, `disaggregated_inference.py` | Single-box inference loops that establish the baseline before disaggregation. |
| `baseline_disaggregated_inference.py`, `optimized_disaggregated_inference.py`, `baseline_prefill_decode_disagg.py`, `optimized_prefill_decode_disagg.py` | Disaggregated pipelines modeling remote prefills, decode overlap, and NVLink pooling. |
| `baseline_kv_cache_management.py`, `optimized_kv_cache_management.py`, `optimized_kv_cache_management_math.py`, `optimized_kv_cache_nvlink_pool.py`, `baseline_kv_cache_local_only.py` | KV-cache orchestration utilities with local-only, math-only, and NVLink-pooled variants. |
| `baseline_continuous_batching.py`, `optimized_continuous_batching.py` | Continuous batching scheduler demonstrating TTFT-aware queueing. |
| `baseline_moe_inference.py`, `optimized_moe_inference.py` | Inference-specific MoE workloads that pair router load with communication control. |
| `baseline_moe_overlap.py`, `optimized_moe_overlap_shared_expert.py`, `baseline_wide_ep.py`, `optimized_wide_ep.py`, `baseline_moe_routing_simple.py`, `optimized_moe_routing_simple_topology_aware.py` | MoE expert-parallel microbenchmarks illustrating overlap, packing/unpacking, and topology-aware routing dispatch. |
| `compare.py`, `requirements.txt`, `expectations_b200.json`, `Makefile` | Harness entry and dependencies for inference-focused validation. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python ch15/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch15
python -m cli.aisp bench run --targets ch15 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_b200.json`; refresh with `--update-expectations` after validating new hardware.

## Demos (Non-Benchmark)
These are runnable parallelism/speculative-decoding companions (not baseline/optimized benchmark pairs). Run them via `aisp demos`:
```bash
python -m cli.aisp demos ch15-tensor-parallel
python -m cli.aisp demos ch15-pipeline-parallel
python -m cli.aisp demos ch15-expert-parallel
python -m cli.aisp demos ch15-context-parallel
python -m cli.aisp demos ch15-speculative-decode
```
- Some demos require `torchrun`/multi-GPU; see per-demo `--help`.
- List all demos: `python -m cli.aisp demos list`
- Pass args to a demo with `--` (e.g., `python -m cli.aisp demos ch15-tensor-parallel -- --help`)

## Validation Checklist
- `python optimized_disaggregated_inference.py --profile minimal` shows reduced fabric stalls compared to the baseline while maintaining accuracy parity.
- `python optimized_kv_cache_management.py --validate` confirms eviction + promotion policies keep decode latency within the budget.
- `python compare.py --examples continuous_batching` proves optimized scheduling increases tokens/sec vs naive queue draining.

## Notes
- `disaggregated_inference.py` can run purely in simulation mode; set `--simulate-network` when hardware isn't wired for NVLink pooling.
- `Makefile` wraps the MPI/UCX targets needed for the multi-node decode experiments.
