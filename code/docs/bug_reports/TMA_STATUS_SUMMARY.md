# TMA Status Summary - Oct 31, 2025

## Current Status

✅ **PRODUCTION READY**: Fallback paths work correctly on all Blackwell hardware  
❌ **TMA BLOCKED**: Descriptor-backed TMA completely non-functional on B200 and GB10

## Hardware Tested

| GPU | Architecture | CUDA SM | TMA Hardware | Driver/Runtime Status |
|-----|--------------|---------|--------------|----------------------|
| B200 | Blackwell | sm_100 | ✅ Supported | ❌ Descriptor API broken |
| GB10 | Grace-Blackwell | sm_121 | ✅ Supported | ❌ Descriptor API broken |

## Test Results (Oct 31, 2025)

### GB10 (SM 12.1) - ARM64/aarch64

```bash
# Build
make ARCH=sm_121 async_prefetch_tma_sm121

# Fallback path (default) - WORKS
./ch7/async_prefetch_tma_sm121
✅ Succeeds via cuda::memcpy_async fallback

# Force TMA path - FAILS
ENABLE_BLACKWELL_TMA=1 ./ch7/async_prefetch_tma_sm121
❌ CUDA_ERROR_INVALID_VALUE from cuTensorMapEncodeTiled (1D)

# 2D pipeline - FAILS
ENABLE_BLACKWELL_TMA=1 ./ch10/tma_2d_pipeline_blackwell
❌ cudaErrorIllegalAddress during TMA kernel sync
```

### B200 (SM 10.0) - x86_64

Same failures as GB10 (see `docs/nvidia_tma_bug_report.md`)

## Root Cause

**Driver/Runtime Bug**: CUDA 13.0 + Driver 580+ cannot handle TMA descriptor operations despite:
1. Hardware reporting `CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 1`
2. Compiler accepting `sm_100` and `sm_121` targets
3. `cuTensorMapEncodeTiled` entry point being available via `cudaGetDriverEntryPointByVersion`

## Workaround

All TMA code is gated behind `ENABLE_BLACKWELL_TMA` environment variable:
- **Default (unset)**: Uses fallback path with `cuda::memcpy_async` ✅
- **Set**: Attempts TMA descriptors (fails on B200/GB10) ❌

## Next Steps for NVIDIA Support

### Information to Share

1. **Updated Bug Report**: `docs/nvidia_tma_bug_report.md`
   - Now includes GB10 (SM 12.1) reproduction steps
   - Confirms issue affects both B200 and GB10 identically

2. **GB10 Logs**: `/tmp/tma_run.log`
   - Full stderr from `ENABLE_BLACKWELL_TMA=1 ./ch10/tma_2d_pipeline_blackwell`
   - Shows `cudaErrorIllegalAddress` during kernel sync

3. **Reproduction Case**: Available in codebase
   - `ch7/async_prefetch_tma.cu` (1D descriptors)
   - `ch10/tma_2d_pipeline_blackwell.cu` (2D descriptors)
   - Helper functions in `common/headers/tma_helpers.cuh` and `common/headers/arch_detection.cuh`

### Questions for NVIDIA

1. Is this a known limitation of CUDA 13.0 with Blackwell B200/GB10?
2. Is there a driver/runtime update in progress?
3. What is the expected timeline for TMA descriptor support?
4. Are there any workarounds other than fallback paths?
5. Will CUDA 13.1 include fixes?

## Impact Assessment

### Performance Impact
- **Memory-bound workloads**: Minimal (TMA vs fallback similar perf)
- **Compute-bound workloads**: Moderate (miss TMA's async benefits)
- **Large pipelines**: Significant (can't use TMA-based optimizations)

### Development Impact
- ✅ Code is production-ready with fallbacks
- ✅ No blocking issues for current workloads
- ❌ Cannot profile TMA kernels with Nsight Compute
- ❌ Cannot optimize with TMA-based patterns
- ❌ Missing potential performance gains

### Timeline
- **Now**: Continue using fallback paths (stable and tested)
- **Q1 2026?**: Monitor for CUDA 13.1 / Driver updates
- **Q2 2026?**: Re-test with updated driver/runtime
- **TBD**: Enable TMA path once fixed

## Recommendations

### For Production
1. ✅ **Keep ENABLE_BLACKWELL_TMA unset** (use fallback)
2. ✅ Run without TMA for all B200/GB10 workloads
3. ✅ Profile fallback paths for optimization opportunities
4. ⏳ Monitor NVIDIA updates for TMA fixes

### For Development
1. ✅ Continue developing with fallback paths
2. ✅ Keep TMA code paths for future enablement
3. ✅ Test regularly with new CUDA/driver releases
4. ⏳ Re-enable TMA once driver fix is available

### For NVIDIA
1. 🔴 **HIGH PRIORITY**: Fix TMA descriptor API for B200/GB10
2. 🔴 Document known limitations in CUDA 13.0 release notes
3. 🟡 Add TMA descriptor tests to driver validation suite
4. 🟡 Provide ETA for fix

## Summary

**The fallback path works perfectly. No immediate action needed for production workloads.**

However, TMA descriptors remain completely broken on all Blackwell hardware tested (B200, GB10), blocking performance optimization opportunities and Nsight profiling.

---

**Last Updated**: October 31, 2025  
**Contact**: See `docs/nvidia_tma_bug_report.md` for full technical details

