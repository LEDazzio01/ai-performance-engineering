# Distributed Training Lab

**Goal**: Scale training across multiple GPUs with FSDP2 and FP8 precision.

## Overview

This lab demonstrates PyTorch's Fully Sharded Data Parallel (FSDP2) with FP8 mixed precision for memory-efficient distributed training on Blackwell GPUs.

## Key Techniques

| Technique | Description | Benefit |
|-----------|-------------|---------|
| FSDP2 | Shard model across GPUs | Linear memory scaling |
| FP8 (E4M3) | 8-bit floating point | 2× memory savings |
| Gradient checkpointing | Recompute activations | 3-4× memory savings |
| Communication overlap | Overlap compute/comm | 1.2-1.5× throughput |

## Files

| File | Description |
|------|-------------|
| `baseline_fsdp2_standard.py` | FSDP2 with BF16 precision |
| `optimized_fsdp2_fp8.py` | FSDP2 with FP8 via torchao |

## Requirements

```bash
pip install torchao  # For FP8 support
```

## Running

```bash
# Single node, 8 GPUs
torchrun --nproc_per_node=8 labs/distributed_training/optimized_fsdp2_fp8.py

# Multi-node (2 nodes, 8 GPUs each)
torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
    labs/distributed_training/optimized_fsdp2_fp8.py
```

## Configuration

```python
# In optimized_fsdp2_fp8.py
benchmark = OptimizedFSDP2FP8(
    batch_size=8,        # Per-GPU batch size
    seq_length=2048,     # Sequence length
    hidden_dim=4096,     # Model hidden dimension
    num_layers=32,       # Transformer layers
    use_fp8=True,        # Enable FP8 training
    use_checkpointing=True,  # Gradient checkpointing
)
```

## Memory Comparison

| Configuration | Memory per GPU | Throughput |
|--------------|----------------|------------|
| Baseline (BF16) | 40 GB | 1.0× |
| + Gradient checkpointing | 15 GB | 0.9× |
| + FP8 | 10 GB | 1.1× |
| **Optimized (all)** | **8 GB** | **1.2×** |

## What to Look For

- **Memory usage**: Use `nvidia-smi` or `torch.cuda.memory_summary()`
- **Communication overhead**: Look for NCCL calls in nsys profiles
- **Compute utilization**: Check SM throughput in ncu

## Related Chapters

- **Ch4**: Distributed computing (NCCL, tensor parallel)
- **Ch13**: PyTorch profiling and FP8
- **Ch13**: FSDP2 and communication patterns

