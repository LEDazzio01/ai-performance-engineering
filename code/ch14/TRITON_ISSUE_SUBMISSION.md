# [SM 10.0] TMA descriptor compilation fails with deep pipelines

## Summary

Compilation crashes in `tritongpu-assign-latencies` pass when using TMA tensor descriptors with aggressive pipeline configurations (num_stages >= 4, BLOCK_K >= 64) on Blackwell GPUs (SM 10.0).

## Environment

- **GPU**: NVIDIA B200 (Compute Capability 10.0)
- **CUDA**: 13.0
- **Triton**: 3.5.0
- **PyTorch**: 2.9.0+cu130
- **OS**: Linux 6.8.0-60-generic

## Description

When using `tl.make_tensor_descriptor()` for TMA-accelerated GEMM operations on Blackwell, the compiler fails during the latency assignment pass if pipeline depth or tile sizes exceed conservative thresholds.

### Error Message

```
error: Failures have been detected while processing an MLIR pass pipeline
note: Pipeline failed while executing [`TritonGPUAssignLatencies` on 'builtin.module' operation]
```

### MLIR Pipeline Context

```
pipeline: "builtin.module(
  ...
  tritongpu-assign-latencies{num-stages=4},
  tritongpu-schedule-loops,
  tritongpu-automatic-warp-specialization{num-stages=4},
  tritongpu-pipeline{dump-intermediate-steps=false num-stages=4},
  ...
)"
```

## Reproduction

### Minimal Reproducer

Attached: `triton_tma_reproducer.py`

Run with:
```bash
python triton_tma_reproducer.py
```

### Configuration That WORKS (Conservative)

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, 
                     num_warps=4, num_stages=1),
    ],
    key=['M', 'N', 'K'],
)
```

**Result**: ✅ Compiles and executes successfully

### Configuration That FAILS (Optimal for Blackwell)

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, 
                     num_warps=16, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
```

**Result**: ❌ Compiler crash in tritongpu-assign-latencies

### Kernel Code (Simplified)

```python
@triton.jit
def tma_gemm_kernel(A_ptr, B_ptr, C_ptr, M, N, K, ...):
    # Create TMA descriptors
    A_desc = tl.make_tensor_descriptor(
        A_ptr, shape=[M, K], strides=[...],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    B_desc = tl.make_tensor_descriptor(
        B_ptr, shape=[K, N], strides=[...],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # GEMM loop with TMA loads
    for k0 in range(0, K, BLOCK_K):
        a = A_desc.load([m0, k0])
        b = B_desc.load([k0, n0])
        acc += tl.dot(a, b, out_dtype=tl.float32)
    
    # Store result
    C_desc.store([m0, n0], acc)
```

## Impact

### Performance Degradation

- **Expected Performance**: 350-400 TFLOPS on B200
- **Actual Performance**: 200-210 TFLOPS (conservative config)
- **Performance Loss**: ~2x slower than hardware capability

### Bandwidth Utilization

- **Expected**: 85-90% of 7.8 TB/s HBM3e bandwidth
- **Actual**: ~60-63% utilization
- **Loss**: ~2-3 TB/s unused bandwidth

### Business Impact

Blackwell B200/B300 GPUs cannot reach advertised performance in Triton-based workloads. Customers paying premium for HBM3e bandwidth and 5th-gen tensor cores see significantly degraded performance.

## Analysis

### Hypothesis

The latency assignment logic may not properly handle:

1. **Register Pressure**: Larger TMA tiles (BLOCK_K=128 vs 32) increase register usage
2. **Pipeline Depth**: Deep pipelines (num_stages=4-5) amplify register requirements
3. **SM 10.0 Specifics**: Blackwell's architectural changes may need special handling

The combination of TMA descriptors + large tiles + deep pipelines exceeds some threshold in the latency assignment pass.

### Threshold Observations

Compilation succeeds when:
- `BLOCK_K <= 32` OR
- `num_stages == 1` OR  
- `num_warps <= 4`

Compilation fails when:
- `BLOCK_K >= 64` AND
- `num_stages >= 4` AND
- `num_warps >= 8`

### Related Components

- `tritongpu-assign-latencies` pass (primary failure point)
- `triton-nvidia-tma-lowering` (TMA descriptor handling)
- `tritongpu-pipeline` (pipeline scheduling)
- SM 10.0 specific code generation

## Workaround

Currently forced to use sub-optimal configuration:

```python
BLOCK_K=32      # Should be 128+ for Blackwell
num_stages=1    # Should be 4-5 for deep pipelines
num_warps=4     # Should be 16 for tensor core saturation
```

This workaround enables compilation but leaves ~2x performance on the table.

## Expected Behavior

The compiler should successfully handle aggressive TMA configurations on Blackwell:

```python
BLOCK_K=128
BLOCK_M=128, BLOCK_N=128
num_stages=4-5
num_warps=16
```

These parameters are optimal for Blackwell's:
- 5th generation tensor cores
- HBM3e bandwidth (7.8 TB/s)
- Large register file per SM

## Additional Context

### Copy Operations vs GEMM

Interestingly, TMA **copy** operations (without `tl.dot`) can use larger tiles:

```python
# This WORKS with larger tiles and deeper pipelines
triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256}, num_warps=16, num_stages=5)
```

This suggests the issue is specifically with TMA + GEMM (dot operations) + deep pipelines, not TMA descriptors in general.

### SM 9.0 (Hopper) Comparison

Similar TMA code works on Hopper (SM 9.0) with aggressive configs. This appears to be Blackwell-specific.

## Attachments

1. **Reproducer**: `triton_tma_reproducer.py` (attached)
2. **Benchmark**: `benchmark_tma_configs.py` (shows performance delta)
3. **Full Error Log**: (see below or run reproducer)

## Questions for Maintainers

1. Is there a register allocation limit we're hitting with TMA + deep pipelines on SM 10.0?
2. Should the latency assignment pass have Blackwell-specific logic?
3. Are there plans to support optimal TMA configs on Blackwell?
4. Is this a known issue with target timeline for fix?

## References

- NVIDIA Blackwell Architecture Whitepaper
- TMA Programming Guide (CUDA 13.0)
- Triton TMA documentation

---

**Priority**: High - Blocking optimal Blackwell performance
**Reproducibility**: 100% on B200 with provided reproducer
**Impact**: Major performance degradation (2x slower than capable)

