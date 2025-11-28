# Async Input Pipeline Lab

**Goal**: Optimize CPU→GPU data transfer to eliminate input bottlenecks during training.

## Overview

This lab demonstrates how to overlap data loading with GPU computation using PyTorch's DataLoader optimizations and CUDA streams.

## Key Techniques

| Technique | Description | Speedup |
|-----------|-------------|---------|
| `pin_memory=True` | Use pinned (page-locked) host memory | 1.5-2× |
| `non_blocking=True` | Async H2D transfers | 1.2-1.5× |
| `num_workers > 0` | Parallel data loading | 2-4× |
| `prefetch_factor` | Pre-load batches in background | 1.1-1.3× |
| Copy stream | Dedicated stream for H2D overlap | 1.2× |

## Files

| File | Description |
|------|-------------|
| `pipeline.py` | Shared helpers: `PipelineConfig`, DataLoader builder, benchmark base |
| `baseline_async_input_pipeline.py` | No overlap: synchronous loading |
| `optimized_async_input_pipeline.py` | Full overlap: pinned memory + async + workers |

## Running

```bash
# Compare baseline vs optimized
python -m cli.aisp bench compare \
    labs.async_input_pipeline.baseline_async_input_pipeline \
    labs.async_input_pipeline.optimized_async_input_pipeline

# Profile with nsys
nsys profile -o async_pipeline python labs/async_input_pipeline/optimized_async_input_pipeline.py
```

## Configuration

```python
from labs.async_input_pipeline.pipeline import PipelineConfig

cfg = PipelineConfig(
    batch_size=16,
    feature_shape=(3, 64, 64),
    dataset_size=64,
    num_workers=4,           # Parallel loading processes
    prefetch_factor=2,       # Batches to prefetch per worker
    pin_memory=True,         # Page-locked memory for fast H2D
    non_blocking=True,       # Async transfers
    use_copy_stream=True,    # Dedicated H2D stream
)
```

## What to Look For

In **nsys** profiles:
- Look for H2D transfers overlapping with kernel execution
- Check for gaps between compute kernels (indicates data starvation)
- Compare "CUDA API" row between baseline and optimized

## Related Chapters

- **Ch2**: Memory hierarchy and transfer types
- **Ch5**: Storage I/O and DataLoader tuning
- **Ch11**: CUDA streams and async execution



