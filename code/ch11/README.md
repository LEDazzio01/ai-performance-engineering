# Chapter 11: CUDA Streams and Concurrency

## Overview

CUDA streams enable concurrent execution of independent operations, dramatically improving GPU utilization and throughput. This chapter teaches you how to use streams effectively, implement stream-ordered memory allocations, and build multi-stream pipelines that overlap computation and data transfer.

## Learning Objectives

After completing this chapter, you can:

- [OK] Create and manage CUDA streams for concurrent execution
- [OK] Overlap kernel execution, H2D, and D2H transfers
- [OK] Use stream-ordered allocators for zero-copy patterns
- [OK] Implement multi-stream pipelines for maximum throughput
- [OK] Measure and optimize stream concurrency
- [OK] Avoid common stream pitfalls (false dependencies, synchronization issues)

## Prerequisites

**Previous chapters**:
- [Chapter 6: CUDA Basics](.[executable]/[file]) - kernel launches
- [Chapter 10: Pipelines](.[executable]/[file]) - async patterns

**Required**: Understanding of asynchronous execution model

## Stream Fundamentals

### What are CUDA Streams?

**Stream**: A sequence of operations that execute in order on the GPU.

**Key property**: Operations in **different** streams can execute concurrently!

```
Default stream (synchronous):
[H2D] → [Kernel 1] → [Kernel 2] → [D2H]  (serial)

Multiple streams (async):
Stream 0: [H2D #0] → [Kernel #0] → [D2H #0]
Stream 1:     [H2D #1] → [Kernel #1] → [D2H #1]
Stream 2:          [H2D #2] → [Kernel #2] → [D2H #2]
          ↑ All can overlap!
```

**Typical speedup**: **2-3x** for independent operations.

---

## Stream Concurrency Patterns

### Pattern 1: Breadth-First Scheduling

**Goal**: Maximize concurrency by issuing all operations before waiting.

```cpp
// Good: Breadth-first
for (int i = 0; i < N; i++) {
    cudaMemcpyAsync(..., streams[i]);  // Issue all H2Ds
}
for (int i = 0; i < N; i++) {
    kernel<<<..., streams[i]>>>(...);  // Issue all kernels
}
for (int i = 0; i < N; i++) {
    cudaMemcpyAsync(..., streams[i]);  // Issue all D2Hs
}
// Maximum overlap!

// Bad: Depth-first
for (int i = 0; i < N; i++) {
    cudaMemcpyAsync(..., streams[i]);
    kernel<<<..., streams[i]>>>(...);
    cudaMemcpyAsync(..., streams[i]);
    // Waits for each stream to complete before next
}
```

### Pattern 2: Hyper-Q Exploitation

**NVIDIA GPU has 128 hardware queues**: Can truly execute 128 independent operations!

```cpp
const int NUM_STREAMS = 32;  // Exploit Hyper-Q
cudaStream_t streams[NUM_STREAMS];

// Create many streams
for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
}

// Launch many small operations
for (int i = 0; i < 1000; i++) {
    small_kernel<<<1, 32, 0, streams[i % NUM_STREAMS]>>>(data[i]);
}
// GPU executes many in parallel!
```

### Pattern 3: Priority Streams

```cpp
cudaStream_t high_priority, low_priority;

// Create streams with priorities
int least_priority, greatest_priority;
cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);

cudaStreamCreateWithPriority(&high_priority, cudaStreamDefault, greatest_priority);
cudaStreamCreateWithPriority(&low_priority, cudaStreamDefault, least_priority);

// High-priority work preempts low-priority
latency_critical_kernel<<<..., high_priority>>>(...);  // Runs first
batch_processing<<<..., low_priority>>>(...);  // Yields to high-priority
```

---

## Performance Analysis

### Measuring Stream Overlap

Use Nsight Systems to visualize concurrency:

```bash
# Profile a CUDA binary (replace with actual binary name)
../.[executable]/profiling/[file] [executable] baseline
nsys-ui ../.[executable]/ch11/your_binary_*.nsys-rep
```

**Look for**:
- [OK] Overlapping kernel execution rows (concurrent streams)
- [OK] H2D during kernel execution (overlap)
- ERROR: Gaps between operations (missed opportunities)

### Stream Efficiency Metrics

| Configuration | Throughput | Stream Efficiency |
|---------------|------------|-------------------|
| No streams | [file] | 0% (serial) |
| 2 streams | [file] | 60% |
| 4 streams | [file] | 77% [OK] |
| 8 streams | [file] | 84% [OK] |
| 16 streams | [file] | 86% [OK] |

**Diminishing returns**: Beyond 8 streams, little benefit (overhead increases).

---

## How to Run All Examples

```bash
cd ch11

# Install dependencies
pip install -r [file]

# Run baseline/optimized comparisons
python3 [script]  # Compares all baseline/optimized pairs

# Build CUDA examples (if available)
make

# Profile examples (replace with actual binary names)
# ../.[executable]/profiling/[file] [executable] baseline
```

---

## Key Takeaways

1. **Streams enable concurrency**: Independent operations in different streams can overlap → 2-3x speedup.

2. **Stream-ordered allocations are faster**: `cudaMallocAsync` avoids device-wide sync → 3-5x faster for frequent allocations.

3. **Breadth-first scheduling maximizes overlap**: Issue all operations before waiting for any.

4. **Hyper-Q enables massive parallelism**: NVIDIA GPU has 128 hardware queues. Use 8-32 streams for best utilization.

5. **Priority streams for latency**: High-priority streams preempt low-priority → Better latency for critical work.

6. **Profile to validate**: Use Nsight Systems to see actual concurrency, not just hope for it.

7. **Diminishing returns after 8 streams**: More streams = more overhead. 8 is usually optimal.

---

## Common Pitfalls

### Pitfall 1: False Dependencies

**Problem**: Using default stream creates implicit dependencies.

```cpp
// Bad: All operations in default stream (serial)
kernel1<<<blocks, threads>>>(data);  // Default stream
kernel2<<<blocks, threads>>>(data);  // Default stream
// Serialized!

// Good: Explicit streams (concurrent if independent)
kernel1<<<blocks, threads, 0, stream1>>>(data1);
kernel2<<<blocks, threads, 0, stream2>>>(data2);
```

### Pitfall 2: Synchronization Too Early

**Problem**: Calling `cudaDeviceSynchronize()` before issuing all work.

```cpp
// Bad:
for (int i = 0; i < N; i++) {
    kernel<<<..., streams[i]>>>(...);
    cudaStreamSynchronize(streams[i]);  // Blocks here!
}

// Good:
for (int i = 0; i < N; i++) {
    kernel<<<..., streams[i]>>>(...);
}
for (int i = 0; i < N; i++) {
    cudaStreamSynchronize(streams[i]);  // Wait all at end
}
```

### Pitfall 3: Pinned Memory for Async Copies

**Problem**: Async `cudaMemcpyAsync` with pageable memory → Silently synchronizes!

**Solution**: Always use pinned memory:
```cpp
float *h_data;
cudaMallocHost(&h_data, size);  // Pinned
cudaMemcpyAsync(d_data, h_data, size, ..., stream);  // Truly async
```

### Pitfall 4: Too Many Streams

**Problem**: Creating 128 streams → Overhead dominates.

**Reality**: 8-16 streams is usually optimal. Beyond that, diminishing returns.

### Pitfall 5: Stream Leaks

**Problem**: Creating streams but never destroying them → Memory leak.

**Solution**: Always destroy streams:
```cpp
cudaStreamCreate(&stream);
// ... use stream ...
cudaStreamDestroy(stream);  // Don't forget!
```

---

## Next Steps

**CUDA Graphs for ultra-low latency** → [Chapter 12: CUDA Graphs](.[executable]/[file])

Learn about:
- Graph capture for repeatable workloads
- Conditional graphs
- Dynamic parallelism
- Sub-microsecond kernel launches

**Back to pipelines** → [Chapter 10: Tensor Cores and Pipelines](.[executable]/[file])

---

## Additional Resources

- **CUDA Streams**: [Programming Guide - Streams](https://[file].com/cuda/cuda-c-programming-guide/[file]#streams)
- **Stream-Ordered Allocations**: [cudaMallocAsync Documentation](https://[file].com/cuda/cuda-runtime-api/[file])
- **Nsight Systems**: [Stream Analysis](https://[file].com/nsight-systems/UserGuide/[file])
- **Hyper-Q**: [Multi-Process Service](https://[file].com/deploy/mps/[file])

---

**Chapter Status**: [OK] Complete
