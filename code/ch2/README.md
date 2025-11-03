# Chapter 2: B200 Hardware Architecture

## Overview

Understanding your hardware is critical for effective optimization. This chapter provides deep insight into the NVIDIA B200/GB200 architecture, helping you make informed decisions about which optimizations have the biggest impact. Learn about the memory hierarchy, NVLink interconnect, and Grace-Blackwell coherency model.

## Learning Objectives

After completing this chapter, you can:

- ✅ Understand B200/GB200 architecture and specifications
- ✅ Measure and validate hardware capabilities (HBM3e bandwidth, NVLink throughput)
- ✅ Identify hardware bottlenecks in your workloads
- ✅ Understand Grace-Blackwell coherency and when to use zero-copy memory
- ✅ Make architecture-aware optimization decisions

## Prerequisites

**Previous chapters**: [Chapter 1: Performance Basics](../ch1/README.md) - profiling fundamentals

**Hardware**: 8x NVIDIA B200 or GB200 GPUs (examples adaptable to other architectures)

## Examples

### 1. `hardware_info.py` - GPU Introspection

**Purpose**: Programmatically query and display GPU specifications.

**What it shows**:
- GPU name, compute capability, memory size
- SM (Streaming Multiprocessor) count
- Memory bandwidth specifications
- Thread block limits

**How to run**:
```bash
python3 hardware_info.py
```

**Expected output (B200)**:
```
GPU 0: NVIDIA B200
  Compute Capability: 10.0  (Blackwell)
  SMs: 148
  Global Memory: 180 GB (HBM3e)
  Memory Bandwidth: 8 TB/s (theoretical)
  Max threads per block: 1024
  Warp size: 32
```

**Key specs to remember**:
- **180 GB HBM3e** (not 192GB - common misconception!)
- **148 SMs** (not 192 - another common error)
- **8 TB/s** theoretical bandwidth (realistically achieve 30-40%)

---

### 2. `nvlink_c2c_p2p_blackwell.cu` - NVLink Bandwidth Test

**Purpose**: Measure peer-to-peer (P2P) bandwidth between GPUs via NVLink.

**What it demonstrates**:
- P2P memory access enablement
- Unidirectional and bidirectional bandwidth measurement
- NVLink vs PCIe comparison

**How to run**:
```bash
make
./nvlink_c2c_p2p_blackwell
```

**Expected output**:
```
Testing P2P bandwidth between GPU 0 and GPU 1
Unidirectional (GPU 0 -> GPU 1): 250 GB/s
Bidirectional (simultaneous):     450 GB/s
```

**Performance targets (B200)**:
- **NVLink 5.0**: 250+ GB/s unidirectional per link
- **Bidirectional**: 450+ GB/s (with proper overlap)
- **AllReduce (NCCL)**: 273.5 GB/s measured (excellent!)

**Why this matters**: Multi-GPU training/inference depends on fast GPU-to-GPU communication. NVLink is 10x+ faster than PCIe.

---

### 3. `gb200_grace_blackwell_coherency.cu` - Grace-Blackwell Coherency

**Purpose**: Demonstrate cache-coherent memory access between Grace CPU and Blackwell GPU.

**What it demonstrates**:
- Zero-copy memory allocation (CPU-GPU coherent)
- Performance comparison: zero-copy vs explicit transfers
- When coherency helps vs hurts

**How to run** (GB200 only):
```bash
make
./gb200_coherency
```

**Expected behavior**:
- **Small transfers (<1MB)**: Zero-copy wins (no transfer overhead)
- **Large transfers (>10MB)**: Explicit H2D wins (dedicated memory faster)

**Key insight**: Grace-Blackwell coherency eliminates small transfer overhead but isn't a replacement for device memory for large data.

---

### 4. `gb200_topology_aware.py` - NUMA Topology

**Purpose**: Query and display NUMA topology for multi-GPU systems.

**What it shows**:
- GPU-to-GPU affinity
- NUMA node assignments
- NVLink connectivity matrix

**How to run**:
```bash
python3 gb200_topology_aware.py
```

**Expected output**:
```
GPU Topology:
  GPU 0 <-> GPU 1: NVLink (9 links)
  GPU 0 <-> GPU 4: NVSwitch
  GPU 0 <-> CPU:   PCIe Gen5
```

**Why this matters**: Affects multi-GPU scaling in Chapter 4. GPUs within same NVSwitch domain communicate faster.

---

## Hardware Architecture Deep Dive

### B200/GB200 Specifications

| Component | Specification | Notes |
|-----------|--------------|-------|
| Architecture | Blackwell (sm_100) | 10.0 compute capability |
| SMs | 148 | ~19,000 CUDA cores |
| Memory | 180 GB HBM3e | NOT 192 GB! |
| Memory Bandwidth | 8 TB/s | Theoretical; 30-40% achievable |
| FP32 | 2000 TFLOPS (sparse) | Dense: ~1000 TFLOPS |
| FP16/BF16 | 2000 TFLOPS | With Tensor Cores |
| FP8 | 4000 TFLOPS | Transformer Engine |
| INT8 | 8000 TOPS | Inference |
| TDP | 1000W | Per GPU |

### Memory Hierarchy

```
        Fastest
           ↑
    [Registers] ← 65,536 per SM
           ↑
  [L1/Shared Memory] ← 256 KB per SM (configurable split)
           ↑
      [L2 Cache] ← 192 MB (shared across SMs)
           ↑
     [HBM3e Memory] ← 180 GB @ 8 TB/s
           ↑
    [NVLink/NVSwitch] ← GPU-GPU: 250+ GB/s
           ↑
      [System RAM] ← CPU-GPU: ~64 GB/s (PCIe Gen5)
        Slowest
```

**Optimization implication**: Minimize trips down the hierarchy. Keep hot data in registers/shared memory (Chapters 7-9).

### Grace-Blackwell (GB200) Additions

**Grace CPU**: 72-core ARM Neoverse V2  
**Chip-to-Chip (C2C) Link**: 900 GB/s coherent interconnect  
**Unified Memory**: CPU and GPU share coherent address space  

**When to use**:
- ✅ Small, frequent CPU-GPU transfers
- ✅ Irregular memory access patterns
- ❌ Large bulk transfers (use explicit H2D)
- ❌ GPU-only compute (standard B200 is fine)

---

## Performance Analysis

### Measuring Your Hardware

Use the common profiling tools:

```bash
# Check GPU specs
python3 hardware_info.py

# Benchmark NVLink
make
./nvlink_c2c_p2p_blackwell

# Verify NVLink with NCCL (Chapter 4)
python3 ../ch4/nccl_benchmark.py
```

### Expected Baselines (B200)

| Metric | Target | Typical Achievable | Excellent |
|--------|--------|-------------------|-----------|
| HBM3e Bandwidth | 8 TB/s | 2.7 TB/s (34%) | 40% |
| FP16 Compute | 2000 TFLOPS | 1000 TFLOPS (50%) | 65% |
| NVLink P2P | 250 GB/s | 230 GB/s (92%) | 95% |
| NVLink AllReduce | N/A | 273 GB/s | Validated ✅ |

**Reality check**: 40-60% of theoretical peak is EXCELLENT for real workloads. Memory-bound ops (most models) won't hit compute peaks.

---

## How to Run All Examples

```bash
cd ch2

# Install Python dependencies
pip install -r requirements.txt

# Query GPU info
python3 hardware_info.py

# Build CUDA examples
make

# Test NVLink bandwidth
./nvlink_c2c_p2p_blackwell

# GB200 only: Test coherency
./gb200_coherency  # Only on GB200 systems

# Check topology
python3 gb200_topology_aware.py
```

---

## Key Takeaways

1. **Know your specs**: B200 has 180 GB (not 192) and 148 SMs (not 192). Using wrong specs leads to incorrect performance expectations.

2. **Memory bandwidth is the bottleneck**: Most models are memory-bound. HBM3e bandwidth (8 TB/s) matters more than compute (2000 TFLOPS) for large models.

3. **40% is great**: Don't expect to hit 100% of theoretical peaks. 40-60% is excellent for real workloads.

4. **NVLink is critical for multi-GPU**: 250+ GB/s vs ~64 GB/s over PCIe. Multi-GPU scaling depends on it (Chapter 4).

5. **Grace-Blackwell coherency is nuanced**: Great for small transfers, not a replacement for device memory.

6. **Architecture awareness informs optimization**: Knowing the memory hierarchy helps you choose which optimizations to prioritize (Chapters 7-10).

---

## Common Pitfalls

### Pitfall 1: Wrong Spec Expectations
**Problem**: Expecting 192 GB memory or 192 SMs based on older or different SKUs.

**Reality**: B200 has 180 GB and 148 SMs. Check your actual hardware!

### Pitfall 2: Comparing to Theoretical Peak
**Problem**: "My code only achieves 30% of peak compute, it must be slow!"

**Reality**: Most workloads are memory-bound. 30-40% of peak is normal and good.

### Pitfall 3: Assuming P2P is Automatic
**Problem**: P2P access must be explicitly enabled with `cudaDeviceEnablePeerAccess()`.

**Solution**: Always enable P2P before multi-GPU work. PyTorch/NCCL do this automatically.

### Pitfall 4: Overusing Zero-Copy on GB200
**Problem**: Using Grace-Blackwell coherency for large data transfers.

**Reality**: Explicit `cudaMemcpy` is faster for bulk transfers (>10MB). Use zero-copy for small, frequent accesses only.

---

## Next Steps

**Continue the journey** → [Chapter 3: System Tuning](../ch3/README.md)

Learn about:
- NUMA binding for optimal CPU-GPU affinity
- System-level tuning for B200/GB200
- Docker/Kubernetes GPU configuration

**Skip ahead to multi-GPU?** → [Chapter 4: Multi-GPU Training](../ch4/README.md)

---

## Additional Resources

- **Blackwell Architecture**: [NVIDIA Blackwell Whitepaper](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- **NVLink Documentation**: [NVLink and NVSwitch](https://www.nvidia.com/en-us/data-center/nvlink/)
- **Grace-Blackwell**: [GB200 Superchip](https://www.nvidia.com/en-us/data-center/grace-blackwell-superchip/)

---

**Chapter Status**: ✅ Complete  
**Last Updated**: November 3, 2025  
**Tested On**: 8x NVIDIA B200 GPUs (B200 SKU, not GB200)

