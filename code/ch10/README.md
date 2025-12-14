# Chapter 10 - Tensor Core Pipelines & Cluster Features

## Summary
Applies tensor-core friendly scheduling on Blackwell: warp specialization, TMA-powered pipelines, persistent kernels, and thread-block clusters with DSMEM and NVLink-C2C awareness.

## Learning Goals
- Use warp specialization and cp.async/TMA to keep tensor cores saturated.
- Prototype persistent matmuls that amortize launch overhead across iterations.
- Exercise thread-block clusters with and without DSMEM to understand hardware limits.
- Combine PyTorch, Triton, and CUDA kernels while keeping expectations synchronized.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_attention.py`, `optimized_attention.py`, `baseline_flash_attention.py`, `optimized_flash_attention.py`, `analyze_scaling.py` | Attention workloads that span eager, fused, and `torch.compile` paths for modern decoder models. |
| `baseline_batch.py`, `optimized_batch.py`, `baseline_matmul.py`, `optimized_matmul.py`, `baseline_matmul_tcgen05.py`, `optimized_matmul_tcgen05.py` | Tensor-core matmul variants demonstrating tcgen05 lowering, register tiling, and PyTorch integration. |
| `baseline_double_buffered_pipeline.{py,cu}`, `optimized_double_buffered_pipeline.{py,cu}`, `baseline_tma_2d_pipeline.py`, `optimized_tma_2d_pipeline.py` | Async pipeline samples mixing cp.async, TMA, and manual double buffering. |
| `baseline_cluster_group*.{py,cu}`, `optimized_cluster_group*.{py,cu}`, `cluster_group_common.cuh`, `cluster_group_utils.py` | Clustered kernel suite covering DSMEM-enabled and DSMEM-free thread-block clusters. |
| `baseline_cooperative_persistent.{py,cu}`, `optimized_cooperative_persistent.{py,cu}`, `baseline_persistent_matmul_tma.py`, `optimized_persistent_matmul_tma.py` | Persistent kernels combining cooperative groups with TMA streams for steady-state throughput. |
| `baseline_flash_attn_tma_micro_pipeline.{py,cu}`, `optimized_flash_attn_tma_micro_pipeline.{py,cu}`, `baseline_warp_specialized_pipeline*.{py,cu}`, `optimized_warp_specialized_pipeline*.{py,cu}` | Micro-pipeline and warp specialization studies that mix Triton, CUDA, and inline PTX. |
| `compare.py`, `workload_config.py`, `perf_sweep.sh`, `profile.sh`, `requirements_cufile.txt` | Harness entry, workload dials, performance sweeps, Nsight automation, and optional cuFile deps. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python ch10/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch10
python -m cli.aisp bench run --targets ch10 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_b200.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- Cluster-enabled kernels fail fast on hardware without DSMEM support, while DSMEM-free variants still execute-use this to confirm cluster capability flags.
- `python optimized_flash_attn_tma_micro_pipeline.py --profile` produces fewer kernel launches and higher achieved FLOP/s than the baseline script.
- `python perf_sweep.sh --chapter ch10` records TFLOP/s vs batch size curves, proving persistent kernels amortize launch costs.

## Notes
- `cufile_gds_example.py` demonstrates integrating GPUDirect Storage into tensor-core pipelines for IO-heavy training loops.
- `requirements_cufile.txt` holds the optional `cufile` wheel; install it only on hosts with GPUDirect Storage enabled.
