# Lab - Full-Stack Blackwell Cluster

## Summary
Demonstrates advanced cluster GEMM kernels (baseline vs DSMEM/TMA optimized) and MoE all-to-all communication readiness probes.

## Learning Goals
- Inspect cluster GEMM kernels (baseline and DSMEM/TMA optimized) via the CUDA extension.
- Validate MoE fabrics with targeted all-to-all sweeps before deploying expert-parallel jobs.
- Track GPU requirements, expected shapes, and automation scripts in one place.

## Benchmarks

### Cluster GEMM (Single-GPU)
Demonstrates optimized matrix multiplication using DSMEM and TMA on Blackwell architecture.
- `baseline_cluster_gemm.py` - Reference GEMM implementation
- `optimized_cluster_gemm.py` - Optimized with DSMEM/TMA (~4x speedup)
- `*_tcgen05.py` variants - For SM100+ with tcgen05 support

### MoE Readiness (Multi-GPU)
All-to-all communication probes for MoE deployment validation. **Requires 2+ GPUs**.
- `baseline_moe_readiness.py` - Basic all-to-all sweep
- `optimized_moe_readiness.py` - With NCCL tuning and heatmap generation

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_cluster_gemm.py`, `optimized_cluster_gemm.py` | Cluster GEMM kernel benchmarks |
| `baseline_moe_readiness.py`, `optimized_moe_readiness.py` | MoE all-to-all probes (multi-GPU) |
| `capstone_extension.py`, `capstone_kernels.cu` | PyTorch extension and CUDA kernels |
| `run_lab_fullstack_cluster.py`, `gpu_requirements.py` | Runner and hardware requirements |

## Running the Benchmarks
```bash
cd ai-performance-engineering

# Cluster GEMM (single-GPU OK)
python -m cli.aisp bench run --targets labs/fullstack_cluster:cluster_gemm

# MoE readiness (requires 2+ GPUs)
python -m cli.aisp bench run --targets labs/fullstack_cluster:moe_readiness

# Direct runner
python labs/fullstack_cluster/run_lab_fullstack_cluster.py --size 2048
```

## Notes
- `gpu_requirements.py` reports minimum GPU count, memory, and features for each benchmark.
- `capstone_extension.py` caches builds under `~/.cache/torch_extensions`.
- MoE readiness defaults assume a GPT-OSS-20B MoE-style shard; adjust flags to match your workload.
