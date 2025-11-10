# Chapter 5: Storage and IO Optimization

## Overview

Even the fastest GPU training can be bottlenecked by slow data loading. This chapter covers storage and IO optimization techniques, including GPUDirect Storage (GDS), efficient DataLoader configuration, and eliminating the IO bottleneck that often limits end-to-end throughput.

## Learning Objectives

After completing this chapter, you can:

- [OK] Identify IO bottlenecks using profiling
- [OK] Implement GPUDirect Storage (GDS) for direct storage-to-GPU transfers
- [OK] Optimize PyTorch DataLoader for maximum throughput
- [OK] Apply prefetching and caching strategies
- [OK] Measure and eliminate IO wait time
- [OK] Configure storage systems for GPU workloads

## Prerequisites

**Previous chapters**:
- [Chapter 1: Performance Basics](.[executable]/[file]) - profiling to identify IO bottlenecks
- [Chapter 3: System Tuning](.[executable]/[file]) - system configuration

**Required**: Fast storage (NVMe SSD recommended, GDS requires specific hardware)

## Examples

###  Traditional IO Optimization

**Purpose**: Demonstrate standard techniques for optimizing data loading without specialized hardware.

**Techniques covered**:

#### Technique 1: Multi-Worker DataLoader
**Problem**: Single-threaded data loading can't keep up with GPU consumption.

**Solution**: Use multiple workers to parallelize loading:
```python
DataLoader(
    dataset,
    batch_size=256,
    num_workers=8,  # Parallel data loading
    pin_memory=True,  # Faster H2D transfers
    prefetch_factor=2,  # Prefetch 2 batches per worker
    persistent_workers=True  # Keep workers alive between epochs
)
```

**Expected impact**: **2-4x faster** data loading (varies by CPU cores and storage speed).

#### Technique 2: Data Prefetching
**Problem**: GPU waits for next batch while current batch computes.

**Solution**: Overlap data loading with compute:
```python
# Prefetch next batch while GPU processes current batch
prefetcher = DataPrefetcher(dataloader)
for batch in prefetcher:
    # batch is already on GPU, ready to use
    output = model(batch)
```

**Expected impact**: **Eliminates IO wait time** if prefetch faster than compute.

#### Technique 3: In-Memory Caching
**Problem**: Reading from disk is slow, especially for small files.

**Solution**: Cache dataset in RAM:
```python
# One-time load into memory
cached_dataset = [data for data in dataset]  # ~100 GB for ImageNet
dataloader = DataLoader(cached_dataset, ...)
```

**Expected impact**: **10-100x faster** than disk (limited by RAM size).

#### Technique 4: Memory-Mapped Files
**Problem**: Large datasets don't fit in RAM.

**Solution**: Use mmap for OS-level caching:
```python
data = [file]('[file]', dtype='float32', mode='r', shape=(N, D))
```

**Expected impact**: First epoch slow, subsequent epochs fast (if data fits in page cache).

**How to run**:
```bash
python3 [script] --workers 8 --prefetch
```

**Expected output**:
```
Configuration: 1 worker, no prefetch
Data loading time: 850 ms/batch
GPU utilization: 45%

Configuration: 8 workers, prefetch=2
Data loading time: 120 ms/batch
GPU utilization: 92% [OK]
```

**Speedup**: **7x faster** data loading, **2x higher** GPU utilization.

---

###  GPUDirect Storage (GDS)

**Purpose**: Demonstrate GPUDirect Storage for direct NVMe-to-GPU transfers, bypassing CPU and system RAM.

**What is GDS?**

Traditional IO path:
```
NVMe SSD → CPU/RAM → GPU  (2 copies, CPU bottleneck)
```

GDS IO path:
```
NVMe SSD → GPU directly  (1 copy, no CPU involvement)
```

**Requirements**:
- NVMe SSD with GDS support
- NVIDIA GPUDirect Storage drivers
- cuFile library
- NVIDIA GPU (or A100/H100 with compatible drivers)

**Benefits**:
- **2-3x faster** for large sequential reads
- **Zero CPU overhead** (CPU free for other work)
- **Lower latency** (eliminates memory copies)

**When to use**:
- [OK] Large files (>10 MB) - amortizes cuFile overhead
- [OK] Sequential access patterns
- [OK] High-throughput workloads (video processing, medical imaging)
- ERROR: Small random reads (overhead too high)
- ERROR: If storage is slower than PCIe (defeats purpose)

**How to run**:
```bash
# Check GDS is available
nvidia-smi nvlink --status

# Install cuFile
pip install cufile-cu12  # Or appropriate CUDA version

# Run example
python3 [script] --file [file] --size 1GB
```

**Expected output**:
```
Traditional IO (CPU path):
  Read 1 GB: 450 ms
  Throughput: [file] GB/s
  CPU usage: 85%

GPUDirect Storage:
  Read 1 GB: 170 ms
  Throughput: [file] GB/s [OK]
  CPU usage: 5%

Speedup: [file], CPU freed for other work!
```

**Realistic speedup**: **2-3x for large files**, minimal benefit for small files.

---

### 3. [source file] – Direct cuFile Read (CUDA 13)

**Purpose**: Provide the smallest possible working cuFile example for systems with GPUDirect Storage enabled.

**What it covers**:
- Uses the CUDA Python bindings (`cuda-python>=[file]`) to call `driver_open`, `handle_register`, `buf_register`, and `read`.
- Reads bytes straight into a CUDA tensor and prints throughput plus a byte preview.
- Falls back from `O_DIRECT` automatically when the filesystem does not support it.

**How to run**:
```bash
# Install dependencies (includes cuda-python and KvikIO bindings)
pip install -r [file]

# Create a 4 MiB sample file and copy it to the GPU via cuFile
python3 [script] /tmp/gds-[file] 4194304 --generate

# Read without O_DIRECT (helpful for network or non-GDS filesystems)
python3 [script] /path/to/[file] 1048576 --no-odirect
```

**Expected output**:
```
Read 4194304 bytes via cuFile in [file] ms ([file] GB/s).
Opened with O_DIRECT: True
Buffer preview: 1f 8b 08 d5 ...
```

**Tip**: Run this example after `gdscheck -p` to verify your cuFile stack before launching larger end-to-end pipelines.

---

## IO Bottleneck Analysis

### Identifying IO Bottlenecks

**Symptoms**:
- Low GPU utilization (<70%)
- High CPU "waiting" time in profiler
- Long gaps between kernel launches

**How to diagnose**:

```bash
# Profile storage I/O optimization
../.[executable]/profiling/[file] [executable].py
../.[executable]/profiling/[file] [executable].py

# Look for in Nsight Systems:
# - Large gaps between CUDA kernels
# - CPU busy in data loading functions
# - High time in DataLoader.__next__()
```

**In Chrome trace**:
```
Timeline view:
  [GPU Kernel] [----gap----] [GPU Kernel] [----gap----]
                 ↑
               IO wait!
```

**If gap > 20% of iteration time** → IO is bottleneck!

### Common IO Bottleneck Causes

| Symptom | Cause | Solution |
|---------|-------|----------|
| GPU utilization <50% | Single-threaded data loading | Increase `num_workers` |
| Gaps between kernels | No prefetching | Enable `prefetch_factor=2` |
| High CPU usage | Too many workers | Reduce workers, optimize preprocessing |
| Slow first epoch | Cold cache | Use in-memory dataset or mmap |
| Network storage slow | Bandwidth limit | Copy dataset to local NVMe |

---

## DataLoader Optimization Checklist

### Optimal Configuration

```python
from [file].data import DataLoader

dataloader = DataLoader(
    dataset,
    
    # Batch size: As large as GPU memory allows
    batch_size=256,
    
    # Workers: Start with num_CPUs / num_GPUs, tune empirically
    num_workers=8,  # For 64-core CPU with 8 GPUs
    
    # Pinned memory: Always enable for CUDA
    pin_memory=True,
    
    # Prefetch: 2-4 batches per worker
    prefetch_factor=2,
    
    # Persistent workers: Avoid respawning overhead
    persistent_workers=True,
    
    # Drop last: For fixed-size batches
    drop_last=True,
    
    # Shuffling: Only if needed (adds overhead)
    shuffle=False  # Use DistributedSampler for multi-GPU
)
```

### Worker Count Tuning

**Too few workers** → GPU starved  
**Too many workers** → CPU overhead, memory pressure

**Rule of thumb**:
```
num_workers = num_CPU_cores / num_GPUs
```

**Example for 8x NVIDIA GPU with 64-core CPU**:
```
num_workers = 64 / 8 = 8 workers per GPU
```

**Tune empirically**: Try 4, 8, 16 and measure GPU utilization.

---

## Storage System Configuration

### NVMe SSD Optimization

For GPU workloads, configure NVMe for high throughput:

```bash
# 1. Use noop or none scheduler (for NVMe)
echo none | sudo tee /sys/block/nvme0n1/queue/scheduler

# 2. Increase read-ahead
echo 8192 | sudo tee /sys/block/nvme0n1/queue/read_ahead_kb

# 3. Disable write barriers (if journaling not critical)
mount -o nobarrier /dev/nvme0n1 /mnt/data

# 4. Use XFS or EXT4 (better for large files than EXT3)
[file] -f /dev/nvme0n1

# 5. Mount with optimal options
mount -o noatime,nodiratime /dev/nvme0n1 /mnt/data
```

**Expected improvement**: **10-20% faster** sequential reads.

### Network Storage (NFS/Lustre)

For shared storage:

```bash
# NFS mount options for GPU training
mount -t nfs -o rsize=1048576,wsize=1048576,timeo=600 \
    server:/export /mnt/data

# Lustre striping for large files
lfs setstripe -c 8 -S 4M /mnt/data  # 8 OSTs, 4MB stripe size
```

**Reality check**: Network storage is 10-100x slower than local NVMe. Always profile first!

---

## Performance Analysis

### Measuring IO Performance

**Baseline measurement**:
```bash
# Measure raw storage bandwidth
fio --name=sequential --rw=read --bs=1M --size=10G --filename=/mnt/data/test

# Expected for NVMe Gen4:
#   Bandwidth: 5-7 GB/s
#   Latency: <100 μs
```

**DataLoader throughput**:
```python
import time
for i, batch in enumerate(dataloader):
    if i == 0:
        start = [file]()
    if i == 100:
        elapsed = [file]() - start
        throughput = 100 * batch_size / elapsed
        print(f"{throughput:.1f} samples/sec")
        break
```

**Target**: DataLoader should saturate GPU (GPU utilization > 90%).

---

## How to Run All Examples

```bash
cd ch5

# Install dependencies
pip install -r [file]

# 1. Traditional IO optimization
python3 [script] \
    --data-dir /mnt/nvme/imagenet \
    --workers 8 \
    --prefetch 2

# 2. GPUDirect Storage (if available)
python3 [script] \
    --file /mnt/nvme/[file] \
    --size 10GB

# 3. Profile data loading
../.[executable]/profiling/[file] [executable].py

# 4. View timeline to identify IO gaps
nsys-ui ../.[executable]/ch5/storage_io_optimization_*.nsys-rep
```

---

## Key Takeaways

1. **IO is often the bottleneck**: Profiling frequently shows GPU waiting for data. Always measure!

2. **Multi-worker DataLoader is essential**: Single-threaded loading can't keep up. Use 4-16 workers depending on system.

3. **Pinned memory is free speedup**: Always enable `pin_memory=True` for CUDA. Gives 2-6x faster H2D transfers.

4. **Prefetching hides latency**: `prefetch_factor=2` overlaps next batch load with current batch compute.

5. **GDS for large files only**: GPUDirect Storage gives 2-3x speedup for large (>10 MB) sequential reads. Not worth it for small files.

6. **Local NVMe >> network storage**: If possible, copy dataset to local NVMe before training (10-100x faster than NFS).

7. **Tune empirically**: Optimal worker count varies by dataset, preprocessing, and hardware. Profile and experiment!

---

## Common Pitfalls

### Pitfall 1: Too Much Image Preprocessing
**Problem**: CPU-heavy augmentation (rotate, crop, color jitter) in DataLoader slows loading.

**Solution**: Use GPU-based augmentation (DALI, Kornia) or simplify transforms.

### Pitfall 2: Reading Many Small Files
**Problem**: ImageNet has [file] JPEG files. Opening files dominates load time.

**Solution**: 
- Repack into large files (TFRecords, LMDB, HDF5)
- Use WebDataset format (tar archives)
- Cache in RAM if possible

### Pitfall 3: Network Storage Bottleneck
**Problem**: 100 Gb Ethernet shared by multiple nodes → only 2-3 GB/s per node.

**Solution**: Copy dataset to local NVMe at job start (one-time cost, persistent benefit).

### Pitfall 4: Forgetting `persistent_workers=True`
**Problem**: Workers respawn every epoch → wasted startup time.

**Solution**: Set `persistent_workers=True` to keep workers alive.

### Pitfall 5: Using GDS for Small Files
**Problem**: cuFile overhead (~50 μs) dominates for small files.

**Solution**: Only use GDS for files >10 MB. Use traditional IO for small files.

---

## Next Steps

**Continue the CUDA journey** → [Chapter 6: Your First CUDA Kernel](.[executable]/[file])

Learn about:
- Writing basic CUDA kernels
- Thread hierarchy and indexing
- Launching kernels from host code
- Basic parallelization patterns

**Back to multi-GPU** → [Chapter 4: Multi-GPU Training](.[executable]/[file])

---

## Additional Resources

- **GPUDirect Storage**: [NVIDIA GDS Documentation](https://[file].com/gpudirect-storage/)
- **cuFile API**: [cuFile API Reference](https://[file].com/cuda/cuda-cufile-api/)
- **DataLoader Optimization**: [PyTorch DataLoader Best Practices](https://[file]/docs/stable/[file]#single-and-multi-process-data-loading)
- **NVIDIA DALI**: [GPU-accelerated data loading](https://[file].com/deeplearning/dali/)

---

**Chapter Status**: [OK] Complete
