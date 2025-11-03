# Chapter 9 Examples Summary

All examples referenced in README.md are now fully implemented and functional.

## Complete Example List

### Python Scripts
1. **`roofline_analysis.py`** ✅ NEW
   - RooflineAnalyzer class for B200
   - Benchmarks vector and matrix operations
   - Plots kernels on roofline model
   - Outputs: roofline_analysis.png
   - Usage: `python3 roofline_analysis.py`

2. **`fusion_pytorch.py`** ✅ EXISTING
   - PyTorch automatic fusion with torch.compile
   - Custom fused operations
   - Usage: `python3 fusion_pytorch.py`

### CUDA Examples

3. **`micro_tiling_matmul.cu`** ✅ NEW
   - Demonstrates tiling impact on arithmetic intensity
   - Three implementations:
     - Naive (AI ~0.25 FLOP/Byte)
     - Tiled 32×32 shared memory (AI ~32 FLOP/Byte)
     - Register-tiled 32×32 + 8×8 (AI ~256 FLOP/Byte)
   - Usage: `./micro_tiling_matmul_sm100`

4. **`arithmetic_intensity_demo.cu`** ✅ NEW
   - Five optimization techniques:
     - Baseline (AI ~0.125 FLOP/Byte)
     - Unrolled (better ILP)
     - Vectorized (float4 loads)
     - Optimized (vectorized + expf, AI ~2.6 FLOP/Byte)
     - Fused (multi-op kernel, AI ~2.0 FLOP/Byte)
   - Usage: `./arithmetic_intensity_demo_sm100`

5. **`fused_l2norm.cu`** ✅ EXISTING
   - Fused L2 normalization kernel
   - Demonstrates memory traffic reduction
   - Usage: `./fused_l2norm_sm100`

6. **`cutlass_gemm_example.cu`** ✅ EXISTING
   - NVIDIA CUTLASS library integration
   - High-performance GEMM
   - Usage: `./cutlass_gemm_example_sm100`

7. **`inline_ptx_example.cu`** ✅ EXISTING
   - Inline PTX assembly for architecture-specific features
   - Usage: `./inline_ptx_example_sm100`

8. **`two_stage_pipeline.cu`** ✅ EXISTING
   - Pipelined kernel fusion with double buffering
   - Usage: `./two_stage_pipeline_sm100`

## Build Status

All examples build successfully:
```bash
make clean && make all
# Output: 6 CUDA executables (sm_100 architecture)
```

## Dependencies

- **requirements.txt**: torch, matplotlib, numpy
- Install: `pip install -r requirements.txt`

## Testing

Quick test of new examples:
```bash
# Roofline analysis (generates plot)
python3 roofline_analysis.py

# Micro-tiling comparison
./micro_tiling_matmul_sm100

# Arithmetic intensity optimizations
./arithmetic_intensity_demo_sm100
```

## README.md Alignment

All examples referenced in ch9/README.md now exist:
- ✅ Line ~109-242: roofline_analysis.py
- ✅ Line ~332-442: micro_tiling_matmul.cu
- ✅ Line ~726-796: arithmetic_intensity_demo.cu
- ✅ Line ~445+: fusion_pytorch.py (existing)
- ✅ Line ~445+: fused_l2norm.cu (existing)
- ✅ Line ~189+: cutlass_gemm_example.cu (existing)
- ✅ Line ~239+: inline_ptx_example.cu (existing)
- ✅ Line ~277+: two_stage_pipeline.cu (existing)

## Chapter 9 Status

**✅ COMPLETE**

The chapter now fully supports the roofline/arithmetic intensity focus from the book:
- Roofline model fundamentals
- Arithmetic intensity measurement
- Micro-tiling optimization
- Kernel fusion (as one technique)
- AI tuning strategies

All referenced examples are implemented, tested, and building successfully.

