# Triton Bug Report Package - Ready for Submission

## Files for Triton GitHub Issue

### 1. **TRITON_ISSUE_SUBMISSION.md** - Main Issue Template
Complete GitHub issue formatted in Markdown with:
- Detailed bug description
- Environment information
- Reproduction steps
- Impact analysis
- Configuration comparisons
- Hypothesis and suggested investigation areas

**Action**: Copy contents directly to GitHub issue

### 2. **triton_tma_reproducer.py** - Minimal Reproducer
Standalone Python script that demonstrates the bug:
- Tests both working (conservative) and failing (aggressive) configurations
- Prints comprehensive environment information
- Provides clear pass/fail results
- Includes all necessary setup (allocator, etc.)

**Action**: Attach to GitHub issue

### 3. **Supporting Documentation**
- `benchmark_tma_configs.py` - Shows performance impact quantitatively
- `TRITON_BUG_INVESTIGATION.md` - Investigation notes
- Error logs from `triton_tma_blackwell.py`

## Quick Test

Run the reproducer to confirm bug on your Blackwell system:

```bash
python ch14/triton_tma_reproducer.py
```

**Expected Output:**
```
Conservative config (BLOCK_K=32, stages=1):  ✅ WORKS
Aggressive config (BLOCK_K=128, stages=4):   ❌ FAILS

✓ BUG CONFIRMED: tritongpu-assign-latencies bug
```

## Bug Summary

**Title**: [SM 10.0] TMA descriptor compilation fails with deep pipelines

**Component**: `tritongpu-assign-latencies` pass

**Symptom**: 
```
error: Failures have been detected while processing an MLIR pass pipeline
note: Pipeline failed while executing [`TritonGPUAssignLatencies` on
      'builtin.module' operation]
```

**Impact**:
- 2x performance loss on Blackwell B200/B300
- ~40% HBM3e bandwidth left unused
- Blocks optimal TMA usage on new architecture

**Configuration That Works**:
```python
BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
num_warps=4, num_stages=1
```

**Configuration That Fails**:
```python
BLOCK_M=128, BLOCK_N=128, BLOCK_K=128  
num_warps=16, num_stages=4
```

## Submission Checklist

- [x] Reproducer script created and tested
- [x] Bug confirmed on Blackwell B200
- [x] Issue template written with full details
- [x] Environment information documented
- [x] Performance impact quantified
- [x] Workaround documented
- [x] Related components identified

## Next Steps

1. **Submit to Triton GitHub**:
   - URL: https://github.com/triton-lang/triton/issues
   - Use `TRITON_ISSUE_SUBMISSION.md` as template
   - Attach `triton_tma_reproducer.py`
   - Tag as: `bug`, `compiler`, `sm_100`, `tma`

2. **Monitor Issue**:
   - Watch for maintainer responses
   - Provide additional info if requested
   - Test any proposed patches

3. **Track Workaround Impact**:
   - Document production performance with conservative configs
   - Plan migration once fix is available
   - Benchmark fix to verify improvement

## Contact

For questions about this bug report:
- See production code: `ch14/triton_tma_blackwell.py`
- Performance analysis: `benchmark_tma_configs.py`
- Investigation notes: `TRITON_BUG_INVESTIGATION.md`

---

**Status**: Ready for submission  
**Priority**: High (blocks Blackwell optimal performance)  
**Reproducibility**: 100% on B200 with provided reproducer  
**Impact**: Major (2x performance degradation)

