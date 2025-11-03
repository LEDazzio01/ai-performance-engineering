# NVIDIA Support Package - TMA Descriptor Issues

**Date**: October 31, 2025  
**Issue**: TMA descriptor APIs non-functional on Blackwell B200 and GB10 with CUDA 13.0

---

## Quick Summary for NVIDIA

**Problem**: `cuTensorMapEncodeTiled` returns `CUDA_ERROR_INVALID_VALUE` (1D) or `cudaErrorIllegalAddress` (2D) on both B200 (SM 10.0) and GB10 (SM 12.1) despite hardware reporting TMA support.

**Reproduction**: See attached bug report with full repro steps for both architectures.

**Impact**: TMA descriptor path completely non-functional; all production code must use fallback paths.

---

## Files to Share with NVIDIA

### 1. Bug Report (PRIMARY)
**File**: `docs/nvidia_tma_bug_report.md`

Contains:
- Detailed issue description
- Environment configurations (B200 and GB10)
- Complete reproduction steps
- Error messages with parameter dumps
- Both 1D and 2D descriptor failures

### 2. GB10 Test Logs
**File**: `/tmp/tma_run.log`

GB10-specific failure log showing:
```bash
ENABLE_BLACKWELL_TMA=1 ./ch10/tma_2d_pipeline_blackwell
# Full stderr trace with cudaErrorIllegalAddress
```

### 3. Source Code (REFERENCE)
**Files**:
- `ch7/async_prefetch_tma.cu` - 1D descriptor test
- `ch10/tma_2d_pipeline_blackwell.cu` - 2D descriptor test
- `common/headers/tma_helpers.cuh` - TMA helper functions with debug logging
- `common/headers/arch_detection.cuh` - Architecture detection utilities

These are self-contained reproducers that can be shared if NVIDIA requests them.

---

## Key Points to Emphasize

### 1. Both B200 and GB10 Affected
- **B200 (SM 10.0)**: Reported earlier, still not fixed
- **GB10 (SM 12.1)**: Identical failures confirmed Oct 31, 2025
- **Same symptoms**: Both architectures fail identically

### 2. Hardware vs Software Mismatch
- Hardware reports `CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 1` ✅
- CUDA 13.0 accepts `sm_100` and `sm_121` targets ✅
- `cuTensorMapEncodeTiled` entry point available ✅
- Descriptor creation/usage fails ❌

### 3. Fallback Works Perfectly
- All code has fallback paths using `cuda::memcpy_async`
- No functional blockers, but:
  - Missing performance optimization opportunities
  - Cannot profile TMA kernels
  - Cannot validate TMA-based algorithms

### 4. Production Impact
**Current state**: Production-ready with fallbacks  
**Future concern**: Cannot leverage TMA optimizations when fixed

---

## Test Results Summary

### GB10 (SM 12.1) - Tested Oct 31, 2025

```bash
# Build
make ARCH=sm_121 async_prefetch_tma_sm121

# Fallback (default) - WORKS
./ch7/async_prefetch_tma_sm121
✅ SUCCESS via cuda::memcpy_async

# TMA path (forced) - FAILS
ENABLE_BLACKWELL_TMA=1 ./ch7/async_prefetch_tma_sm121
❌ CUDA_ERROR_INVALID_VALUE from cuTensorMapEncodeTiled

# 2D pipeline - FAILS
ENABLE_BLACKWELL_TMA=1 ./ch10/tma_2d_pipeline_blackwell
❌ cudaErrorIllegalAddress during kernel execution
```

**Full log**: `/tmp/tma_run.log`

### B200 (SM 10.0) - Previously Tested

Same failures (see `docs/nvidia_tma_bug_report.md` for details)

---

## Questions for NVIDIA

1. **Is this a known issue?**
   - Any internal tracking number we can reference?

2. **Timeline for fix?**
   - Which driver/runtime version will include the fix?
   - Is CUDA 13.1 expected to resolve this?

3. **Workaround available?**
   - Any undocumented flags or settings we should try?
   - Should we avoid TMA entirely for now?

4. **Root cause?**
   - Driver issue, runtime issue, or hardware/software handshake?
   - Why does hardware report support but operations fail?

5. **Testing support?**
   - Can we get beta drivers for testing?
   - How can we help validate the fix?

---

## Contact & Repository

**Repository**: Local codebase available for inspection if needed  
**Hardware access**: Can run any additional tests on B200 and GB10  
**Availability**: Ready to test fixes as soon as available

---

## Additional Context

### Why This Matters

TMA (Tensor Memory Accelerator) is a key Blackwell feature for:
- Asynchronous data movement (hiding memory latency)
- Multi-level pipeline optimization
- Complex tensor transformations in hardware

Without working TMA descriptors:
- Must use software-based async copies (cuda::memcpy_async)
- Missing hardware acceleration benefits
- Cannot validate TMA-optimized algorithms

### Current Mitigation Strategy

All TMA code is **production-ready with fallbacks**:
1. Check `ENABLE_BLACKWELL_TMA` environment variable
2. If unset → use fallback (default, works perfectly)
3. If set → attempt TMA (currently fails, for testing only)

This means:
- ✅ No production blockers
- ✅ Code ready to enable TMA when fixed
- ✅ Can validate fix by setting environment variable

---

## Appendix: File Locations

All files relative to project root:

```
docs/
├── nvidia_tma_bug_report.md      # Primary bug report
├── TMA_STATUS_SUMMARY.md         # Executive summary
├── NVIDIA_SUPPORT_PACKAGE.md     # This file
└── README.md                     # Documentation index

ch7/
└── async_prefetch_tma.cu         # 1D descriptor reproducer

ch10/
└── tma_2d_pipeline_blackwell.cu  # 2D descriptor reproducer

common/headers/
├── tma_helpers.cuh               # TMA helper functions (logging)
└── arch_detection.cuh            # Architecture detection

/tmp/tma_run.log                  # GB10 test output (stderr)
```

---

**Next Step**: Share `docs/nvidia_tma_bug_report.md` and `/tmp/tma_run.log` with NVIDIA support team.

**Status**: Ready to test any proposed fixes on both B200 and GB10 hardware.

