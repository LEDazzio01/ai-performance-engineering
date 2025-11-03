# NVIDIA Support Ticket – Blackwell TMA Descriptor Failures

## Summary

- **Issue 1 – cuTensorMapEncodeTiled (1D)**: Returns `CUDA_ERROR_INVALID_VALUE` on Blackwell B200 and GB10 when encoding a 1‑D FP32 tensor map that matches the CUDA 13.0 programming guide requirements (aligned base pointer, 16‑byte stride, 1024‑element box).
- **Issue 2 – cuTensorMapEncodeTiled (2D)**: Encoding succeeds for a 4096×4096 FP32 tensor, but the first `cp_async_bulk_tensor_2d_*` transfer triggers `cudaErrorIllegalAddress`.

These regressions prevent the CUDA 13.0 TMA demos from running and block Nsight Compute profiling pipelines.

## Environment

### B200 Configuration (Original Report)
- GPU: 8× NVIDIA B200 (SM 10.0) – confirmed via `nvidia-smi -L`
- CUDA Toolkit / Driver: 13.0
- Nsight Systems / Nsight Compute: 2024.1
- Operating System: Ubuntu 22.04 (Lambda reference image)

### GB10 Configuration (Updated Oct 31, 2025)
- GPU: NVIDIA GB10 (Grace-Blackwell, SM 12.1) – confirmed via `nvidia-smi -L`
- CUDA Toolkit / Driver: 13.0, Driver 580+
- CUDA Architecture: `sm_121`
- Operating System: Ubuntu 22.04 / ARM64 (aarch64)

## Reproduction Steps

### B200 (SM 10.0)

1. Build the demos:
   ```bash
   ./scripts/build_tma_demos.sh --arch sm_100
   ```
2. Run the 1‑D prefetch sample (fails immediately):
   ```bash
   ./ch7/async_prefetch_tma
   ```
   Output:
   ```
   [TMA] cuTensorMapEncodeTiled (1D) failed: invalid argument
          dataType=7 (FLOAT32)
          rank=1
          base=0x70d2df200000 / 0x70d2df240000
          elements=65536
          stride_bytes=4
          box=1024
          elem_stride=1
          swizzle=0
          l2=2 (CU_TENSOR_MAP_L2_PROMOTION_L2_128B)
   ```
3. Run the 2‑D pipeline (descriptor encodes, kernel aborts):
   ```bash
   ./ch10/tma_2d_pipeline_blackwell
   ```
   Log:
   ```
   CUDA error (warm-up sync): an illegal memory access was encountered
   ```

### GB10 (SM 12.1) – Updated Oct 31, 2025

1. Build the demos:
   ```bash
   make ARCH=sm_121 async_prefetch_tma_sm121
   ```
2. Run without TMA (fallback path works correctly):
   ```bash
   ./ch7/async_prefetch_tma_sm121
   # ✅ Succeeds via cuda::memcpy_async fallback
   ```
3. Force TMA path (fails identically to B200):
   ```bash
   ENABLE_BLACKWELL_TMA=1 ./ch7/async_prefetch_tma_sm121
   # ❌ CUDA_ERROR_INVALID_VALUE from cuTensorMapEncodeTiled (1D)
   ```
4. Test 2D pipeline:
   ```bash
   ENABLE_BLACKWELL_TMA=1 ./ch10/tma_2d_pipeline_blackwell
   # ❌ cudaErrorIllegalAddress during TMA kernel sync
   ```

Both B200 and GB10 tests use device pointers returned by `cudaMalloc`, tensor ranks/strides/boxes identical to the CUDA 13.0 guide (§10.29), and the helper prints confirm the exact parameter values passed to `cuTensorMapEncodeTiled`.

## Attachments / References

- `ch7/async_prefetch_tma.cu`
- `ch10/tma_2d_pipeline_blackwell.cu`
- Logging implemented in `common/headers/tma_helpers.cuh` (functions: `make_1d_tensor_map`, `make_2d_tensor_map`) to dump descriptor parameters.

## Request

Please investigate the driver/runtime handling of `cuTensorMapEncodeTiled` on **both Blackwell B200 (SM 10.0) and GB10 (SM 12.1)** for the configurations above. If additional traces or a reduced repro are required, let us know.

### Key Observations

1. **Hardware supports TMA**: Both B200 and GB10 report `CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 1`
2. **Driver recognizes architecture**: CUDA 13.0 accepts `sm_100` and `sm_121` targets
3. **Descriptor creation fails**: `cuTensorMapEncodeTiled` returns `CUDA_ERROR_INVALID_VALUE` for 1D descriptors
4. **2D descriptors create but fail**: Descriptors encode successfully but kernel execution hits `cudaErrorIllegalAddress`
5. **Fallback path works**: `cuda::memcpy_async` and `cuda::pipeline` work correctly without TMA descriptors

### Impact

- TMA descriptor path is **completely non-functional** on Blackwell GB10/B200 with CUDA 13.0
- All production code must use fallback paths (gated behind `ENABLE_BLACKWELL_TMA` env var)
- Nsight Compute cannot profile TMA-based kernels
- Performance optimization opportunities blocked

### Logs Available

- Full stderr trace from GB10 2D pipeline test available in `/tmp/tma_run.log`
- Can provide additional traces, CUDA-MEMCHECK output, or reduced reproducers on request

Thanks! A fallback path is in place for now (`ENABLE_BLACKWELL_TMA` env var), but we'd like to restore the descriptor-backed pipeline once a fixed driver/runtime is available.

---

**Last Updated**: October 31, 2025  
**Status**: Regression confirmed on both B200 (SM 10.0) and GB10 (SM 12.1) with CUDA 13.0 + Driver 580+
