# KV Cache Optimization Lab

**Goal**: Reduce KV cache memory footprint using FP8/FP4 quantization for longer context lengths.

## Overview

This lab demonstrates KV cache compression techniques that enable serving longer sequences without running out of GPU memory. Essential for production LLM inference.

## Key Techniques

| Technique | Compression | Quality Impact |
|-----------|-------------|----------------|
| FP16 baseline | 1× | None |
| FP8 (E4M3) | 2× | Minimal (<0.1% perplexity) |
| NVFP4 | 4× | Small (<0.5% perplexity) |
| Dynamic scaling | - | Preserves accuracy |

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Module exports |
| `baseline_kv_standard.py` | FP16 KV cache (standard) |
| `optimized_kv_fp8_compressed.py` | FP8/FP4 compressed KV cache |

## Running

```bash
# Compare memory usage
python -m tools.cli.benchmark_cli compare \
    labs.kv_optimization.baseline_kv_standard \
    labs.kv_optimization.optimized_kv_fp8_compressed

# Profile memory
python labs/kv_optimization/optimized_kv_fp8_compressed.py
```

## Configuration

```python
benchmark = OptimizedKVFP8Compressed(
    batch_size=8,
    num_layers=32,
    num_heads=32,
    head_dim=128,
    max_seq_length=8192,  # Context length
    use_fp8=True,         # 2× compression
    use_fp4=False,        # 4× compression (more aggressive)
)
```

## Memory Savings

For a 70B model with 8K context:

| Precision | KV Cache Size | Max Batch |
|-----------|---------------|-----------|
| FP16 | 32 GB | 4 |
| FP8 | 16 GB | 8 |
| FP4 | 8 GB | 16 |

## Implementation Details

### FP8 Quantization
```python
# Per-layer dynamic scaling
scale = x.abs().max() / 448.0  # E4M3 max value
x_fp8 = (x / scale).to(torch.float8_e4m3fn)

# Dequantization
x_restored = x_fp8.to(torch.float16) * scale
```

### Block-wise Scaling
For better accuracy, use block-wise scaling (128 elements per block):
```python
x_blocks = x.view(-1, 128)
scales = x_blocks.abs().amax(dim=-1, keepdim=True) / 448.0
x_fp8 = (x_blocks / scales).to(torch.float8_e4m3fn)
```

## What to Look For

- **Memory reduction**: Compare `torch.cuda.memory_allocated()` before/after
- **Accuracy**: Monitor perplexity or downstream task metrics
- **Throughput**: Quantization adds overhead; ensure net speedup

## Related Chapters

- **Ch15**: KV cache management and PagedAttention
- **Ch19**: NVFP4 and dynamic precision switching
- **Ch18**: Speculative decoding (benefits from longer context)

