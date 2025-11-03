# Performance Optimization Quick Start Guide

This guide helps you apply the optimizations from the profiling analysis to your own code.

---

## 🚀 Quick Wins (5 minutes)

### 1. Enable Pinned Memory in PyTorch
```python
# In your training script:
train_loader = DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True,  # Required for non_blocking H2D copies
    num_workers=4
)

# Pair with non-blocking transfers from pinned buffers:
data = data.to(device, non_blocking=True)
```
**Impact:** 2-6x faster CPU↔GPU transfers[^impact-pinned] (requires pinned+non_blocking copies)

---

### 2. Preallocate Tensors
```python
# BAD - Allocates every iteration (210ms overhead):
for batch in dataloader:
    x = batch['input'].to(device)
    
# GOOD - Preallocate once:
data_buf = torch.empty(batch_size, *input_shape, device=device)
target_buf = torch.empty(batch_size, device=device, dtype=torch.long)

for batch in dataloader:
    data_buf.copy_(batch["input"], non_blocking=True)
    target_buf.copy_(batch["target"], non_blocking=True)
```
**Impact:** Eliminates 210ms CPU overhead from `aten::empty_strided`[^impact-prealloc]

---

### 3. Use Batched Operations
```python
# BAD - 40 separate GEMM calls:
for i in range(40):
    output[i] = torch.matmul(A[i], B[i])

# GOOD - Single batched call:
output = torch.bmm(A, B)  # Uses cublasSgemmStridedBatched
```
**Impact:** 31x faster by collapsing 40 GEMM launches into one[^impact-batched]

---

### 4. Increase Batch Size
```python
# Try larger batches to saturate GPU:
batch_sizes = [32, 64, 128, 256]

# Expected improvement:
# Batch 32:  11K MFLOPs
# Batch 256: 95K MFLOPs (8.5x better)
```
**Impact:** 8.5x more GEMM efficiency at batch 256[^impact-batchsize]

---

## 🔧 Kernel Optimizations (CUDA)

### 1. Add Launch Bounds
```cuda
// Hint to compiler for better register allocation:
__global__ void __launch_bounds__(256, 8) my_kernel(...) {
    // 256 threads per block
    // Min 8 blocks per SM target
}
```
**Impact:** Better occupancy/register tradeoff[^impact-launch]

---

### 2. Vectorized Memory Access
```cuda
// BAD - Scalar loads:
float val = data[idx];

// GOOD - Vector loads (128-bit transactions):
int idx4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
if (idx4 + 3 < n) {
    float4 vec = *reinterpret_cast<float4*>(&data[idx4]);
    // Process vec.x, vec.y, vec.z, vec.w
    *reinterpret_cast<float4*>(&data[idx4]) = vec;
}
```
**Impact:** 1.28x memory bandwidth improvement[^impact-vectorized]

---

### 3. Restrict Pointers
```cuda
// Tell compiler pointers don't alias:
__global__ void kernel(float* __restrict__ out,
                       const float* __restrict__ in) {
    // Enables more aggressive optimizations
}
```

---

## 🎯 Advanced Optimizations

### CUDA Graphs (Static Workloads)
```python
# Capture static computation graph:
graph = torch.cuda.CUDAGraph()

with torch.cuda.graph(graph):
    output = model(input_buf)
    loss = criterion(output, target_buf)

# Replay in loop (much faster):
for batch in dataloader:
    input_buf.copy_(batch['input'])
    graph.replay()
```
*Note:* Example captures forward pass and loss only; handle optimizer state updates before graph replay (see `ch1/performance_basics_optimized.py:70`).

**Impact:** Up to 195x faster in static microbenchmarks when launch overhead dominates[^impact-graphs]; expect ~1.1-2x end-to-end once integrated

---

### Shared Memory Prefetching
```cuda
__global__ void __launch_bounds__(256, 8) kernel(float* data, ...) {
    __shared__ float smem[256 * 4];
    
    // Load to shared memory
    float4 vec = *reinterpret_cast<const float4*>(&data[idx]);
    *reinterpret_cast<float4*>(&smem[tid * 4]) = vec;
    __syncthreads();
    
    // Process from shared (better cache locality)
    vec = *reinterpret_cast<float4*>(&smem[tid * 4]);
    // ... compute ...
    
    // Write back
    *reinterpret_cast<float4*>(&data[idx]) = vec;
}
```
**Impact:** 1.24x speedup, enables TMA on Blackwell[^impact-shared]

---

### NVSHMEM & Symmetric Memory Playbook

- `ch4/nvshmem_training_example.py` showcases gradient bucket fusion,
  hybrid FSDP sharding, and pipeline handoff. Use
  `pytest tests/test_nvshmem_training.py -q` to validate the fallback
  paths locally (Gloo backend) before moving to 8x B200 clusters.
- `ch16/symmetric_memory_inference.py` implements distributed KV cache,
  multi-model pooling, and speculative decoding on symmetric memory. Run
  `pytest tests/test_symmetric_memory_inference.py -q` for CPU/Gloo
  verification, then integrate the module via
  `InferenceServer8GPU(..., use_symmetric_kv=True)` for production.
- Compile and experiment with the new CUDA samples for lock-free queues,
  double-buffer pipelines, and multi-node hierarchies:
  ```bash
  cd ch4
  make nvshmem  # builds nvshmem_pipeline_patterns and nvshmem_multinode_example
  ```
- Launch multi-node experiments with
  `./scripts/nvshmem_launch_multinode.sh <nodes> <gpus_per_node> <binary>`.
  This wrapper wires up `nvshmemrun` with the appropriate world size and
  hostfile hints.

Collect performance baselines with
`torchrun --nproc_per_node=8 ch4/nvshmem_vs_nccl_benchmark.py` and stash
the JSON summaries under `profiles/` for comparison with NCCL-only runs.

---

## 📊 Profiling Checklist

### Before Optimizing
```bash
# Profile to identify bottlenecks:
ncu --set speed-of-light --export profile.ncu-rep ./binary
nsys profile --trace cuda --output profile.nsys-rep ./binary

# Export metrics:
ncu --import profile.ncu-rep --csv --page raw > metrics.csv
nsys stats --report cuda_kern_exec_sum profile.nsys-rep
```

### Check These Metrics
- [ ] SM Throughput < 50% → Optimize compute
- [ ] Memory Throughput < 50% → Optimize memory access
- [ ] High CPU time in allocation → Preallocate tensors
- [ ] Many small kernel launches → Batch operations
- [ ] Low MFLOPs → Increase batch size

---

## 🎓 Examples in This Repo

### Working Code Examples
```
ch1/batched_gemm_example.cu          - 31x GEMM speedup demo
ch1/performance_basics_optimized.py  - All PyTorch optimizations
ch11/basic_streams.cu                - Kernel optimization examples
```

### Full Documentation
```
profiles_*/reports/performance_analysis_recommendations.md
profiles_*/reports/implementation_summary.md
```

---

## 📈 Expected Improvements

| Optimization | Typical Speedup | Effort |
|--------------|----------------|--------|
| Pinned memory | 2-6x transfers[^impact-pinned] | 5 min |
| Preallocate tensors | 1.5-2x total[^impact-prealloc] | 10 min |
| Batched operations | 10-30x launch[^impact-batched] | 15 min |
| Larger batch size | 2-8x GEMM[^impact-batchsize] | 5 min |
| Vectorized loads | 1.2-1.5x BW[^impact-vectorized] | 30 min |
| CUDA Graphs | 1.1-2x total (195x microbenchmarks[^impact-graphs]) | 1 hour |

**Cumulative: 5-10x end-to-end speedup**

---

## 🔍 Common Issues

### Issue: "CUDA out of memory"
**Solution:** Increase batch size gradually, use gradient accumulation
```python
# Accumulate gradients over multiple small batches
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Issue: "Kernel takes longer with optimization"
**Solution:** Check if optimization is appropriate
- Shared memory helps for reuse, not simple operations
- Vectorization requires aligned data
- Profile before/after to verify improvement

### Issue: "CUDA Graphs fail to capture"
**Solution:** Graphs require static shapes
- All tensors must have same size each iteration
- No dynamic control flow during capture
- Use regular execution for dynamic workloads

---

## 📚 Further Reading

### Blackwell-Specific Optimizations
- TMA (Tensor Memory Accelerator) for async copy
- FP8 precision for inference (2x speedup)
- NVLS for multi-GPU collectives

### Tools
- `ncu` - Nsight Compute (kernel profiling)
- `nsys` - Nsight Systems (system-level profiling)
- `torch.profiler` - PyTorch profiling

### Key Docs
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Batched Operations](https://docs.nvidia.com/cuda/cublas/)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

## 🎯 Recommended Order

1. ✅ Profile first (identify bottlenecks)
2. ✅ Quick wins (pinned memory, preallocation)
3. ✅ Batched operations (big impact, easy)
4. ✅ Batch size tuning (test multiple sizes)
5. ✅ Kernel optimizations (if compute-bound)
6. ✅ CUDA Graphs (for static workloads)
7. ✅ Re-profile (verify improvements)

**Start with profiling, apply quick wins, measure improvement!**

[^impact-pinned]: Measured with pinned DataLoader and non_blocking copies in `profiles_manual_20251028_063656/reports/implementation_summary.md:132`.
[^impact-prealloc]: PyTorch profiler showed 210 ms `aten::empty_strided` overhead in `profiles_manual_20251028_063656/reports/performance_analysis_recommendations.md:11`; elimination confirmed in `implementation_summary.md:107-111`.
[^impact-batched]: Batched GEMM benchmark results recorded in `profiles_manual_20251028_063656/reports/implementation_summary.md:184-190`.
[^impact-batchsize]: Batch-size sweep metrics captured in `profiles_manual_20251028_063656/reports/implementation_summary.md:207-215`.
[^impact-launch]: Launch-bound guidance implemented in `profiles_manual_20251028_063656/reports/implementation_summary.md:15-26`.
[^impact-vectorized]: Vectorized kernel bandwidth improvement measured in `profiles_manual_20251028_063656/reports/implementation_summary.md:45-52`.
[^impact-shared]: Shared-memory prefetch speedup measured in `profiles_manual_20251028_063656/reports/implementation_summary.md:55-79`.
[^impact-graphs]: Static CUDA Graph capture goodput improvement documented in `profiles_manual_20251028_063656/reports/implementation_summary.md:156-160`.
