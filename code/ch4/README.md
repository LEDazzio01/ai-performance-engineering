# Chapter 4 - Multi-GPU Distribution

## Summary
Demonstrates how to scale training and inference across multiple Blackwell GPUs with NVLink/NVSwitch fabric awareness, NCCL tuning, NVSHMEM collectives, and symmetric memory patterns.

## Learning Goals
- Benchmark data-parallel and tensor-parallel training loops with and without overlap.
- Quantify NVLink bandwidth and topology effects when mixing local and disaggregated GPUs.
- Experiment with NVSHMEM pipelines to reduce host involvement in GPU synchronization.
- Adopt symmetric memory pools to simplify KV-cache replication and optimizer state sharding.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_dataparallel.py`, `optimized_dataparallel.py`, `baseline_dataparallel_basic.py`, `optimized_dataparallel_basic.py` | Reference data-parallel loops that compare naive gradient exchange vs fused+overlapped NCCL. |
| `baseline_no_overlap.py`, `optimized_no_overlap.py`, `baseline_no_overlap_basic.py`, `optimized_no_overlap_basic.py` | Overlap studies that stage compute/comm concurrency and pipeline microbatches to hide allreduce latency. |
| `baseline_nvlink.py`, `optimized_nvlink.py`, `baseline_nvlink_multigpu.py`, `optimized_nvlink_multigpu.py`, `baseline_nvlink_topology_blind.py`, `optimized_nvlink_topology_aware.py` | NVLink and NVSwitch exercises for validating peer bandwidth and NUMA-aware rank placement. |
| `baseline_continuous_batching_multigpu.py`, `optimized_continuous_batching_multigpu.py`, `baseline_disaggregated.py`, `optimized_disaggregated.py` | Continuous batching + disaggregated inference demos that showcase NVLink pooling and remote KV reuse. |
| `baseline_nvshmem_pipeline_parallel_multigpu.py`, `optimized_nvshmem_pipeline_parallel_multigpu.py`, `baseline_nvshmem_training_example_multigpu.py`, `optimized_nvshmem_training_example_multigpu.py` | NVSHMEM pipeline and training samples highlighting device-driven synchronization benefits. |
| `baseline_symmetric_memory_multigpu.py`, `optimized_symmetric_memory_multigpu.py`, `baseline_symmetric_memory_perf.py`, `optimized_symmetric_memory_perf.py` | Symmetric memory utilities for distributed KV cache and optimizer shards. |
| `compare.py`, `requirements.txt`, `expectations_gb10.json`, `bandwidth_benchmark_suite_multigpu.py`, `nccl_benchmark.py` | Harness driver plus standalone NCCL/NVLink sweepers for topology bring-up. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ch4
python compare.py --profile none
python tools/cli/aisp bench list-targets --chapter ch4
python tools/cli/aisp bench run --targets ch4 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_gb10.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python compare.py --examples dataparallel` shows the optimized pair overlapping compute and communication with lower latency.
- `python bandwidth_benchmark_suite_multigpu.py --profile minimal` surfaces >=250 GB/s links on connected GPU pairs and highlights any slow hops.
- NVSHMEM samples emit consistent outputs when `NVSHMEM_SYMMETRIC_SIZE` is sized to hold the workload; mismatched config raises clear errors.

## Notes
- `symmetric_memory_*` helpers hold user-space allocators for pooling KV-cache lines across GPUs without NVSwitch penalties.
- Use `nccl_blackwell_config.py` to seed NCCL env vars (min NRings, IB mapping) before launching multi-node tests.
- `baseline_nvshmem_ibgda_microbench.py` / `optimized_nvshmem_ibgda_microbench.py` wrap the C++ IBGDA microbenchmark; run with `python tools/cli/aisp bench run --targets ch4:nvshmem_ibgda_microbench --profile none` once NVSHMEM is installed.

### NVSHMEM IBGDA (GPUDirect Async) quick reference
- Why: lets SMs ring NIC doorbells directly, removing the CPU proxy; blog data shows up to 9.5× higher throughput for sub-1 KiB puts and ~180 MOPS for register-level `nvshmem_p`.
- Enable (InfiniBand + NVSHMEM 2.7+):
  ```bash
  export NVSHMEM_IB_ENABLE_IBGDA=1
  export NVSHMEM_IBGDA_NIC_HANDLER=gpu
  export NVSHMEM_IBGDA_FORCE_NIC_BUF_MEMTYPE=gpumem
  # optional: NVSHMEM_IBGDA_ENABLE_MULTI_PORT=1 NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH=1
  # optional: NVSHMEM_DEBUG=INFO NVSHMEM_INFO=1
  ```
- Try it here: compare NCCL vs NVSHMEM symmetric memory with/without IBGDA:
  ```bash
  cd /mnt/dev-fin-03/ai-performance-engineering/code
  NVSHMEM_IB_ENABLE_IBGDA=1 NVSHMEM_DEBUG=INFO NVSHMEM_INFO=1 \
    torchrun --nproc_per_node=8 ch4/nvshmem_vs_nccl_benchmark.py \
    --min-bytes 1024 --max-bytes 1048576 --steps 4
  ```
- Expect NVSHMEM columns to improve for ≤16 KiB payloads when IBGDA is active; if not, verify NVSHMEM version, IB firmware/driver, and GPUDirect RDMA.
