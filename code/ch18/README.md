# Chapter 18: Advanced Attention Mechanisms

## Overview

Attention is the core bottleneck in transformer inference and training. This chapter covers advanced attention optimizations including FlashAttention, FlexAttention for flexible patterns, and MLA (Multi-head Latent Attention) for reduced KV cache size.

## Learning Objectives

After completing this chapter, you can:

- [OK] Implement FlashAttention for memory-efficient attention
- [OK] Use FlexAttention for document masking and custom patterns
- [OK] Apply MLA to reduce KV cache by 4-8x
- [OK] Optimize sliding window and block-diagonal attention
- [OK] Profile attention kernels and identify bottlenecks
- [OK] Choose the right attention variant for your use case

## Prerequisites

**Previous chapters**:
- [Chapter 7: Memory Access Patterns](.[executable]/[file]) - coalescing fundamentals
- [Chapter 10: Tensor Cores](.[executable]/[file]) - matrix operations

**Required**: Understanding of attention mechanism and memory hierarchy

---

## Attention Fundamentals

### Standard Attention (Naive)

```python
def naive_attention(Q, K, V):
    # Q, K, V: [batch, heads, seq_len, head_dim]
    scores = [file](Q, [file](-2, -1))  # [batch, heads, seq_len, seq_len]
    scores = scores / [file]([file](-1))
    attention = [file](scores, dim=-1)      # Problem: Materializes huge matrix!
    output = [file](attention, V)
    return output
```

**Problem**: `scores` matrix is `[seq_len, seq_len]` → **O(N²) memory**!
- For seq_len=4096: 4096² × 2 bytes (FP16) = **32 MB per head**
- With 32 heads: **1 GB just for attention scores**!

---

## Examples

###  FlashAttention-style Optimized Attention

**Purpose**: Memory-efficient attention using tiling and recomputation.

**Key idea**: Never materialize full attention matrix. Compute in blocks.

```python
import torch
from flash_attn import flash_attn_func

def flashattention(q, k, v, causal=False):
    """
    FlashAttention: O(N²) time, O(N) memory (vs O(N²) naive).
    
    Args:
        q, k, v: [batch, seq_len, num_heads, head_dim]
        causal: Apply causal masking
    
    Returns:
        output: [batch, seq_len, num_heads, head_dim]
    """
    output = flash_attn_func(
        q, k, v,
        dropout_p=[file],
        softmax_scale=None,  # Auto: 1/sqrt(head_dim)
        causal=causal,
    )
    return output

# Benchmark
batch_size, seq_len, num_heads, head_dim = 32, 4096, 32, 128

q = [file](batch_size, seq_len, num_heads, head_dim, dtype=[file], device='cuda')
k = [file](batch_size, seq_len, num_heads, head_dim, dtype=[file], device='cuda')
v = [file](batch_size, seq_len, num_heads, head_dim, dtype=[file], device='cuda')

# Naive attention
[file].reset_peak_memory_stats()
start = [file]()
output_naive = naive_attention(q, k, v)
[file].synchronize()
time_naive = [file]() - start
memory_naive = [file].max_memory_allocated() / 1e9

# FlashAttention
[file].reset_peak_memory_stats()
start = [file]()
output_flash = flashattention(q, k, v)
[file].synchronize()
time_flash = [file]() - start
memory_flash = [file].max_memory_allocated() / 1e9

print(f"Naive Attention:")
print(f"  Time: {time_naive * 1000:.2f} ms")
print(f"  Memory: {memory_naive:.2f} GB")

print(f"\nFlashAttention:")
print(f"  Time: {time_flash * 1000:.2f} ms ({time_naive / time_flash:.2f}x faster)")
print(f"  Memory: {memory_flash:.2f} GB ({memory_naive / memory_flash:.2f}x less)")
```

**Expected results (seq_len=4096)**:
```
Naive Attention:
  Time: 145 ms
  Memory: [file] GB

FlashAttention:
  Time: 38 ms ([file] faster) [OK]
  Memory: [file] GB (7x less) [OK]
```

**How to run**:
```bash
pip install flash-attn --no-build-isolation
python3 [script]
```

---

###  FlexAttention for Custom Patterns

**Purpose**: Flexible attention patterns (document masking, sliding window, etc.).

**FlexAttention** (PyTorch [file]+): Define custom masking functions in Python!

```python
import torch
from [file].[file]_attention import flex_attention, create_block_mask

def document_masking(b, h, q_idx, kv_idx):
    """
    Mask to prevent attention across document boundaries.
    
    Example: Two docs in one sequence
      Doc 1: tokens 0-1023
      Doc 2: tokens 1024-2047
    Token 512 should only attend to 0-1023, not 1024-2047.
    """
    # Document boundary at 1024
    doc_boundary = 1024
    
    # Same document?
    q_doc = q_idx // doc_boundary
    kv_doc = kv_idx // doc_boundary
    
    return q_doc == kv_doc

def sliding_window_mask(b, h, q_idx, kv_idx, window_size=256):
    """Sliding window attention (local context)."""
    return abs(q_idx - kv_idx) <= window_size

# Create block mask (compiled for efficiency)
block_mask = create_block_mask(document_masking, None, None, Q=q, K=k)

# Apply FlexAttention
output = flex_attention(q, k, v, block_mask=block_mask)
```

**Use cases**:
- **Document masking**: Multi-document batches
- **Sliding window**: Local attention ([file]., Longformer)
- **Block-diagonal**: Separate sequences in batch
- **Custom patterns**: Any pattern you can express in Python!

**How to run**:
```bash
python3 [script]
```

---

### 3. `[CUDA file]` (see source files for implementation) - Multi-head Latent Attention

**Purpose**: Reduce KV cache size with latent compression.

**MLA (DeepSeek-V3)**: Compress K, V to lower-dimensional latent space.

**Standard attention**:
- K, V: [batch, seq_len, num_heads, head_dim]
- KV cache: `2 × seq_len × num_heads × head_dim × 2 bytes`
- For 32 heads, head_dim=128, seq_len=4096: **64 MB per sequence**

**MLA**:
- Compress K, V to latent: [batch, seq_len, latent_dim]
- Latent_dim = 512 (vs num_heads × head_dim = 4096)
- KV cache: `2 × seq_len × latent_dim × 2 bytes`
- For seq_len=4096: **8 MB per sequence** → **8x reduction!**

```cuda
// [file] - MLA attention kernel

__global__ void mla_attention_kernel(
    const half* Q,           // [batch, heads, seq_len, head_dim]
    const half* K_latent,    // [batch, seq_len, latent_dim]
    const half* V_latent,    // [batch, seq_len, latent_dim]
    const half* K_proj,      // [heads, latent_dim, head_dim] - Projection to heads
    const half* V_proj,      // [heads, latent_dim, head_dim]
    half* output,            // [batch, heads, seq_len, head_dim]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int latent_dim
) {
    // Thread/block indexing
    int batch_idx = [file];
    int head_idx = [file];
    int q_idx = [file] * [file] + [file];
    
    if (q_idx >= seq_len) return;
    
    // Load Q for this position
    half q[HEAD_DIM];
    for (int d = 0; d < head_dim; d++) {
        q[d] = Q[batch_idx * num_heads * seq_len * head_dim +
                  head_idx * seq_len * head_dim +
                  q_idx * head_dim +
                  d];
    }
    
    // Attention scores
    float max_score = -INFINITY;
    float sum_exp = [file];
    float scores[MAX_SEQ_LEN];
    
    // For each KV position
    for (int kv_idx = 0; kv_idx <= q_idx; kv_idx++) {  // Causal
        // Project K_latent to this head's K
        half k[HEAD_DIM] = {0};
        for (int d = 0; d < head_dim; d++) {
            for (int l = 0; l < latent_dim; l++) {
                half k_lat = K_latent[batch_idx * seq_len * latent_dim +
                                       kv_idx * latent_dim + l];
                half k_p = K_proj[head_idx * latent_dim * head_dim +
                                   l * head_dim + d];
                k[d] += __hmul(k_lat, k_p);
            }
        }
        
        // Compute score: Q · K
        float score = [file];
        for (int d = 0; d < head_dim; d++) {
            score += __half2float(__hmul(q[d], k[d]));
        }
        score /= sqrtf((float)head_dim);
        
        // Track max for numerical stability
        max_score = fmaxf(max_score, score);
        scores[kv_idx] = score;
    }
    
    // Softmax
    for (int kv_idx = 0; kv_idx <= q_idx; kv_idx++) {
        scores[kv_idx] = expf(scores[kv_idx] - max_score);
        sum_exp += scores[kv_idx];
    }
    for (int kv_idx = 0; kv_idx <= q_idx; kv_idx++) {
        scores[kv_idx] /= sum_exp;
    }
    
    // Weighted sum of V
    half output_vec[HEAD_DIM] = {0};
    for (int kv_idx = 0; kv_idx <= q_idx; kv_idx++) {
        // Project V_latent to this head's V
        half v[HEAD_DIM] = {0};
        for (int d = 0; d < head_dim; d++) {
            for (int l = 0; l < latent_dim; l++) {
                half v_lat = V_latent[batch_idx * seq_len * latent_dim +
                                       kv_idx * latent_dim + l];
                half v_p = V_proj[head_idx * latent_dim * head_dim +
                                   l * head_dim + d];
                v[d] += __hmul(v_lat, v_p);
            }
        }
        
        // Accumulate weighted
        for (int d = 0; d < head_dim; d++) {
            output_vec[d] += __hmul(__float2half(scores[kv_idx]), v[d]);
        }
    }
    
    // Store output
    for (int d = 0; d < head_dim; d++) {
        output[batch_idx * num_heads * seq_len * head_dim +
               head_idx * seq_len * head_dim +
               q_idx * head_dim +
               d] = output_vec[d];
    }
}
```

**Benefits**:
- **4-8x KV cache reduction**
- **Same quality** (trained end-to-end)
- **Slightly slower** (projection overhead), but memory savings enable larger batches

**How to run**:
```bash
make
[executable]
```

---

### 4. `[CUDA file]` (see source files for implementation) - Sliding Window Attention

**Purpose**: Local attention for long sequences (Longformer-style).

```cuda
__global__ void sliding_window_attention_kernel(
    const half* Q, const half* K, const half* V,
    half* output,
    int seq_len, int head_dim, int window_size
) {
    int q_idx = [file] * [file] + [file];
    if (q_idx >= seq_len) return;
    
    // Attention only within window
    int kv_start = max(0, q_idx - window_size);
    int kv_end = min(seq_len, q_idx + window_size + 1);
    
    // Standard attention, but only for [kv_start, kv_end)
    // ... (similar to MLA kernel, but limited range) ...
}
```

**Complexity**: O(N × W) instead of O(N²), where W = window size.

**How to run**:
```bash
make
[executable]
```

---

### 5. `[CUDA file]` (see source files for implementation) - PagedAttention (vLLM)

**Purpose**: Efficient KV cache management with paging.

```cuda
// PagedAttention: KV cache stored in fixed-size blocks (pages)

struct PageTable {
    int* physical_blocks;  // Maps logical block → physical block
    int num_blocks;
};

__global__ void paged_attention_kernel(
    const half* Q,              // [num_seqs, num_heads, head_dim]
    const half* K_cache,        // [num_blocks, block_size, num_heads, head_dim]
    const half* V_cache,        // [num_blocks, block_size, num_heads, head_dim]
    const PageTable* page_tables,  // Per-sequence page table
    half* output,               // [num_seqs, num_heads, head_dim]
    int num_seqs,
    int num_heads,
    int head_dim,
    int block_size
) {
    int seq_idx = [file];
    int head_idx = [file];
    
    PageTable page_table = page_tables[seq_idx];
    
    // Load Q
    half q[HEAD_DIM];
    for (int d = 0; d < head_dim; d++) {
        q[d] = Q[seq_idx * num_heads * head_dim + head_idx * head_dim + d];
    }
    
    // Iterate through logical blocks (pages)
    float max_score = -INFINITY;
    float sum_exp = [file];
    float scores[MAX_BLOCKS * BLOCK_SIZE];
    
    int total_tokens = [file]_blocks * block_size;
    for (int logical_block = 0; logical_block < [file]_blocks; logical_block++) {
        int physical_block = [file]_blocks[logical_block];
        
        // Iterate through tokens in this block
        for (int token_in_block = 0; token_in_block < block_size; token_in_block++) {
            int kv_idx = logical_block * block_size + token_in_block;
            
            // Load K from physical block
            half k[HEAD_DIM];
            for (int d = 0; d < head_dim; d++) {
                k[d] = K_cache[physical_block * block_size * num_heads * head_dim +
                                token_in_block * num_heads * head_dim +
                                head_idx * head_dim + d];
            }
            
            // Compute score
            float score = [file];
            for (int d = 0; d < head_dim; d++) {
                score += __half2float(__hmul(q[d], k[d]));
            }
            scores[kv_idx] = score;
            max_score = fmaxf(max_score, score);
        }
    }
    
    // Softmax and weighted sum (similar to previous kernels)
    // ...
}
```

**Benefits**:
- **No fragmentation**: Fixed-size blocks
- **Easy swapping**: Swap blocks to CPU/NVMe
- **Memory sharing**: Multiple sequences can share pages (prefix caching)

---

## Attention Variants Comparison

| Variant | Memory | Speed | Use Case |
|---------|--------|-------|----------|
| **Naive** | O(N²) | Baseline | Short sequences (<1K) |
| **FlashAttention** | O(N) | 2-4x | General purpose (1K-16K) |
| **FlexAttention** | O(N) | 2-4x | Custom patterns (documents, sliding) |
| **MLA** | O(N × L) | [file] | Large batches (KV cache limited) |
| **Sliding Window** | O(N × W) | 3-5x | Very long sequences (>16K) |
| **PagedAttention** | O(N) | 2-3x | Production serving (memory mgmt) |

---

## How to Run All Examples

```bash
cd ch18

# Install dependencies
pip install -r [file]
pip install flash-attn --no-build-isolation

# FlashAttention-style optimized attention
python3 [script]

# FlexAttention with custom patterns
python3 [script]

# Custom CUDA kernels
make
[executable]
[executable]
[executable]
```

---

## Key Takeaways

1. **FlashAttention is essential**: 2-4x faster, 5-10x less memory. Use by default.

2. **FlexAttention for custom patterns**: Document masking, sliding windows, etc. Easy to implement in Python.

3. **MLA reduces KV cache 4-8x**: Critical for large batch inference.

4. **PagedAttention for production**: Efficient memory management, no fragmentation.

5. **Sliding window for very long sequences**: O(N) complexity for 16K+ tokens.

6. **Profile to choose**: Different variants excel in different scenarios.

7. **Memory vs speed trade-off**: MLA slower but enables larger batches → Higher throughput.

---

## Common Pitfalls

### Pitfall 1: Using Naive Attention for Long Sequences
**Problem**: O(N²) memory → OOM at 8K+ tokens.

**Solution**: Always use FlashAttention for seq_len > 1K.

### Pitfall 2: Not Using Causal Masking
**Problem**: Attention sees future tokens → Leakage!

**Solution**: Enable `causal=True` in FlashAttention for autoregressive models.

### Pitfall 3: Inefficient Custom Masks
**Problem**: Custom mask implementation not optimized → Slow.

**Solution**: Use FlexAttention's compiled masks for custom patterns.

### Pitfall 4: Over-Allocating KV Cache
**Problem**: Allocating max_seq_len for all sequences → Wasted memory.

**Solution**: Use PagedAttention or dynamic allocation.

---

## Next Steps

**Batched operations** → [Chapter 19: Batched GEMM](.[executable]/[file])

Learn about:
- Batched matrix multiplications
- Grouped GEMM for MoE
- cuBLASLt for optimized batching

**Back to profiling** → [Chapter 17: Dynamic Routing](.[executable]/[file])

---

## Additional Resources

- **FlashAttention**: [Paper](https://[file]/abs/[file]), [FlashAttention-2](https://[file]/abs/[file])
- **FlexAttention**: [PyTorch Blog](https://[file]/blog/flexattention/)
- **MLA**: [DeepSeek-V3 Paper](https://[file]/abs/[file])
- **PagedAttention**: [vLLM Paper](https://[file]/abs/[file])

---

**Chapter Status**: [OK] Complete
