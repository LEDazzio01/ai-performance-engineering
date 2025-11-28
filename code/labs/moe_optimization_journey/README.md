# MoE Optimization Journey

A comprehensive lab demonstrating **7 levels** of MoE optimizations with real compound speedups, achieving **~35x** on NVIDIA B200.

## Results Summary (8K tokens, 8 experts)

| Level | Technique | Time | Speedup | Δ from Prev | What It Does |
|-------|-----------|------|---------|-------------|--------------|
| 0 | **Naive** | 2310 ms | 1.0x | baseline | Python loops over experts |
| 1 | **+ Batched** | 503 ms | **4.6x** ✅ | +4.6x | Einsum parallelizes all tokens |
| 2 | **+ Fused** | 482 ms | 4.8x | +4% | Triton fuses SiLU*up |
| 3 | **+ MemEfficient** | 382 ms | **6.0x** ✅ | +26% | Eliminate intermediate tensors |
| 4 | **+ Grouped** | 68 ms | **33.8x** ✅✅✅ | +460% | Per-expert GEMM on sorted tokens |
| 5 | + CUDAGraphs | 68 ms | 33.8x | +0% | (GPU already saturated) |
| 6 | **+ Compiled** | 67 ms | **34.6x** ✅✅✅ | +2% | torch.compile polish |

## Key Insights

### Why PyTorch MatMuls Are Already Fast

When you call `x @ w` in PyTorch, here's what happens:

```
Python operator @  →  torch.matmul()  →  ATen dispatcher  →  CUDA backend  →  cuBLAS GEMM
                                                                              ↑
                                                                 NVIDIA's hand-tuned library!
```

**cuBLAS is already highly optimized** - it uses:
- Tensor Core operations (BF16/FP8)
- Optimal tiling and memory access patterns
- Hardware-specific tuning for each GPU architecture

This means **Python-level micro-optimizations have minimal impact** once you're using tensor operations correctly.

### The Real Bottleneck: Memory Access Pattern

At production scale (8K+ tokens), the performance story is:

| Optimization | Small (2K) | Large (8K) | Why? |
|--------------|------------|------------|------|
| Batched vs Naive | +4.6x | +4.6x | Eliminates Python loops |
| Grouped vs Batched | **+1.5x** | **+7x** | Fixes O(N×E) gather! |
| Everything else | ~0-5% | ~0-5% | MatMuls already optimal |

**The "batched" approach gathers weights for EVERY token:**
```python
w1_sel = w1[expert_indices]  # [N, top_k, H, I] = O(N × top_k × H × I) memory!
```

At 16K tokens, this tries to allocate **128 GB** and OOMs!

**The "grouped" approach only needs O(N × H) memory:**
```python
for e in range(num_experts):
    tokens_e = sorted_tokens[offset:offset+counts[e]]  # Just the tokens for expert e
    output[offset:offset+count] = tokens_e @ w1[e]      # Single cuBLAS call
```

### What Doesn't Help (And Why)

1. **FP8 on-the-fly conversion**: The conversion overhead (6x slower!) outweighs any compute benefit. Native FP8 weights work, but require quantization during model loading.

2. **Multi-stream parallelism**: At 8K+ tokens, each per-expert matmul already saturates the GPU. Adding streams just adds synchronization overhead.

3. **Triton fused activations**: SiLU takes <1% of runtime - fusing it saves ~4%. The matmuls dominate.

4. **torch.compile on loops**: Compile struggles with Python loops over variable-sized tensors. Better to apply at a higher level or use CUDA graphs.

## The Techniques

### Level 1: Batched Execution (4.6x)
```python
# Instead of looping, gather expert weights and use einsum
w1_sel = w1[expert_indices]  # [batch, top_k, h, i]
gate = torch.einsum('bkh,bkhi->bki', x_exp, w1_sel)
```

### Level 4: Grouped GEMM (33.8x) - THE BIG WIN
```python
# Sort tokens by expert for contiguous memory access
flat_idx = expert_indices.view(-1)
sorted_order = torch.argsort(flat_idx, stable=True)
sorted_tokens = tokens.repeat_interleave(K, dim=0)[sorted_order]
counts = torch.bincount(flat_idx[sorted_order], minlength=E)

# Per-expert GEMM on sorted tokens - each is a single cuBLAS call!
offset = 0
for e in range(num_experts):
    tokens_e = sorted_tokens[offset:offset+counts[e]]
    output[offset:offset+counts[e]] = tokens_e @ w1[e]
    offset += counts[e]
```

### Level 6: torch.compile (34.6x)
```python
model = torch.compile(model, mode="max-autotune")  # Adds kernel fusion on top
```

## Advanced: What Would Help Further?

### CUTLASS GroupedGEMM
Instead of 8 separate cuBLAS calls (one per expert), CUTLASS can fuse them into a single kernel:
- Eliminates kernel launch overhead
- Better SM utilization
- Used by vLLM/SGLang for production MoE

**Status**: vLLM's `fused_experts()` provides this, but requires hardware-specific config files (missing for B200).

### Expert Tensor Parallelism  
Split experts across multiple GPUs:
- Expert 0-3 on GPU 0
- Expert 4-7 on GPU 1
- AllReduce to combine results

**Status**: Multi-stream parallelism on single GPU doesn't help because grouped GEMM already saturates the GPU.

### Native FP8 Weights
Store weights in FP8 format at model load time, not converting on-the-fly:
```python
# Bad: Convert every forward pass
w1_fp8 = w1.to(torch.float8_e4m3fn)  # Adds overhead!

# Good: Store in FP8 from the start using torch._scaled_mm
# Requires column-major layout: w_cm = w.T.contiguous()
result = torch._scaled_mm(x_fp8, w_cm.T, scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)
```

**Status**: Works but requires careful handling of column-major layout for `_scaled_mm`. No speedup observed at 8K scale because FP8 conversion overhead matches compute savings.

### Why Further Optimizations Are Hard

At 8K+ tokens with our grouped implementation:
1. **Each expert matmul is already cuBLAS-optimal** (~200+ TFLOPS)
2. **Python loop overhead is ~10µs per iteration** (negligible vs 2ms matmul)
3. **GPU is saturated** - no idle SMs to parallelize further

The **real gains** require:
- **CUTLASS GroupedGEMM** - fuse all expert matmuls into ONE kernel (used by vLLM)
- **Tensor parallelism** - split across GPUs for linear scaling
- **Kernel fusion** - fuse activation + matmul (what torch.compile does)

## Running the Benchmarks

```bash
# Using aisp bench CLI
python -m cli.aisp bench run -t moe_journey

# Run individual levels
python labs/moe_optimization_journey/level0_naive.py
python labs/moe_optimization_journey/level1_batched.py
python labs/moe_optimization_journey/level2_fused.py
python labs/moe_optimization_journey/level3_memefficient.py
python labs/moe_optimization_journey/level4_grouped.py
python labs/moe_optimization_journey/level5_cudagraphs.py
python labs/moe_optimization_journey/level6_compiled.py

# Run all levels with timing
for level in 0 1 2 3 4 5 6; do
  python labs/moe_optimization_journey/moe_benchmark.py $level
done
```

## Files

| File | Description |
|------|-------------|
| `level0_naive.py` | Baseline: Sequential expert execution |
| `level1_batched.py` | + Batched einsum |
| `level2_fused.py` | + Triton fused SiLU*up kernel |
| `level3_memefficient.py` | + Memory efficient ops |
| `level4_grouped.py` | + Per-expert grouped GEMM |
| `level5_cudagraphs.py` | + CUDA graphs |
| `level6_compiled.py` | + torch.compile (fully optimized) |
| `moe_model.py` | Configurable MoE with all techniques |
| `moe_benchmark.py` | Base benchmark class |
| `triton_kernels.py` | Custom Triton kernels |

## Scale Matters!

The optimizations show different benefits at different scales:

| Tokens | Batched | Grouped | Speedup |
|--------|---------|---------|---------|
| 2K | 24 ms | 16 ms | 1.5x |
| 8K | 93 ms | 16 ms | **5.8x** |
| 16K | 185 ms | 16 ms | **11.4x** |
| 32K | **OOM!** | 17 ms | **∞** |

**Always test with realistic production workloads!**

## The Bottom Line

For MoE optimization:

1. **Architecture > Micro-optimization**: Changing HOW you compute (grouped vs batched) beats tweaking existing code
2. **Know your bottleneck**: Profile before optimizing - the matmuls are already cuBLAS-optimal
3. **Scale reveals truth**: Small workloads hide the real bottlenecks
4. **When in doubt, torch.compile**: It captures many small wins automatically

## Hardware

- **GPU**: NVIDIA B200 (Blackwell)
- **CUDA**: 13.0
- **PyTorch**: 2.7+

## References

Based on techniques from:
- **ch15**: Expert parallelism, MoE overlap
- **ch19**: Token bucketing, MXFP8 quantization
- **vLLM/SGLang**: CUTLASS grouped GEMM, FP8 native weights
