# Chapter 11 - Streams & Concurrency

## Summary
Explains how to overlap compute, memory, and communication on Blackwell using CUDA streams, ordered sequences, Hyper-Q, warp-specialized pipelines, and adaptive scheduling.

## Learning Goals
- Use multiple CUDA streams to overlap independent kernels without starving priority work.
- Control ordering constraints for KV-cache updates and stream-ordered memory pools.
- Benchmark warp-specialized multistream kernels that share data via DSMEM.
- Introduce adaptive policies that adjust stream usage based on runtime telemetry.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_streams.py`, `optimized_streams.py`, `baseline_streams.cu`, `optimized_streams_ordered.cu`, `stream_overlap_base.py` | Core stream overlap demos that contrast serialized launches with overlapped workloads. |
| `baseline_stream_ordered.py`, `baseline_stream_ordered_kv_cache.py`, `optimized_stream_ordered.py`, `optimized_stream_ordered_kv_cache.py` | Stream-ordered allocator and KV-cache examples ensuring deterministic updates while enabling overlap. |
| `baseline_gemm_streams.py`, `optimized_gemm_streams.py`, `baseline_tensor_cores_streams.py`, `optimized_tensor_cores_streams.py` | GEMM pipelines that schedule tensor-core kernels across multiple streams to decouple math vs IO phases. |
| `baseline_distributed_streams.py`, `optimized_distributed_streams.py`, `baseline_adaptive_streams.py`, `optimized_adaptive_streams.py` | Adaptive streaming controllers that balance NCCL, compute, and IO tasks on large systems. |
| `baseline_warp_specialization_multistream.*`, `optimized_warp_specialized_multistream.*`, `warp_specialized_cluster_pipeline_multistream.cu` | Warp-specialized multistream kernels demonstrating DSMEM usage and per-stream specialization. |
| `compare.py`, `Makefile`, `expectations_gb10.json` | Harness driver plus expectation data for concurrency regressions. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ch11
python compare.py --profile none
python tools/cli/aisp bench list-targets --chapter ch11
python tools/cli/aisp bench run --targets ch11 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_gb10.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python optimized_streams.py --trace` captures overlapping NVTX ranges in Nsight Systems, proving concurrency is active.
- `python optimized_stream_ordered_kv_cache.py --validate` matches the baseline's outputs while reducing idle gaps between cache updates.
- Warp-specialized multistream kernels flag unsupported hardware (missing DSMEM) immediately, preventing silent fallbacks.

## Notes
- `warp_specialized_triton.py` provides a Triton analogue for the CUDA concurrency demos so you can compare compiler-generated schedules.
- `optimized_kv_prefetch_pipeline_enhanced.cu` builds on the DSMEM kernels bundled in this directory so you can study the entire pipeline locally.
