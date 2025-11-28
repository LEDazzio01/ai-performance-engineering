# Third-Party Dependency Notes

## CUTLASS Version Management

### The Problem

TransformerEngine bundles CUTLASS as a git submodule, but their pinned version lags behind
the latest CUTLASS releases. This causes issues on newer hardware like NVIDIA Blackwell (SM100a).

| Component | Version | Notes |
|-----------|---------|-------|
| `third_party/cutlass` | 4.3.0 | Standalone, full SM100a support |
| TransformerEngine's bundled CUTLASS | 4.2.0 | Missing SM100a headers |

### Critical SM100a Headers (Blackwell)

CUTLASS 4.3.0+ includes these headers required for Blackwell:
- `cute/arch/tmem_allocator_sm100.hpp` - TMEM allocator
- `cute/arch/mma_sm100_umma.hpp` - UMMA (Unified MMA)
- `cute/atom/copy_traits_sm100.hpp` - TMA copy traits
- `cutlass/gemm/collective/sm100_mma_array_warpspecialized.hpp`

### The Solution

`setup.sh` replaces TE's bundled CUTLASS with a symlink to our standalone 4.3.0:

```bash
rm -rf "${TE_SRC_DIR}/3rdparty/cutlass"
ln -s "${CUTLASS_SRC_DIR}" "${TE_SRC_DIR}/3rdparty/cutlass"
```

### Verification

Always verify after setup:

```bash
# Quick check
make verify-cutlass

# Full verification
python tools/verification/verify_cutlass_setup.py

# System-wide validation
./tools/verification/validate_system.sh
```

### Version Pinning (setup.sh)

```bash
# These versions are tested together - update carefully!
CUTLASS_REF="8cd5bef43a2b0d3f9846b026c271593c6e4a8e8a"  # 4.3.0
CUTLASS_TARGET_VERSION="4.3.0"
TE_GIT_COMMIT="f8cb598c9f3af2bc512a051abec75590b25f54c4"
```

### Updating Versions

When updating TransformerEngine or CUTLASS:

1. Check TE's `.gitmodules` for their bundled CUTLASS commit
2. Look up that commit's version at `include/cutlass/version.h`
3. Ensure our standalone CUTLASS is >= TE's bundled version
4. Re-run `make verify-cutlass` after setup

### Why We Need TransformerEngine

Despite the complexity, TE provides critical functionality:
- FP8 training/inference (`fp8_autocast`)
- NVFP4/MXFP8 quantization
- Fused FP8 Linear layers
- cuDNN fused attention paths

Used in: `ch16/`, `ch19/`, `labs/nanochat/`, `labs/ultimate_moe_inference/`,
`tools/benchmarking/benchmark_peak.py`, and 230+ other files.

### Troubleshooting

**Symptom**: Build fails with "missing header: tmem_allocator_sm100.hpp"

**Fix**:
```bash
cd third_party
rm -rf TransformerEngine/3rdparty/cutlass
ln -s ../../cutlass TransformerEngine/3rdparty/cutlass
```

**Symptom**: TE builds but crashes on Blackwell

**Check**: Ensure the symlink points to CUTLASS 4.3.0+:
```bash
ls -la TransformerEngine/3rdparty/cutlass
cat cutlass/include/cutlass/version.h | grep CUTLASS_
```




