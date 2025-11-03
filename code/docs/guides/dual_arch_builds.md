# Dual-Architecture Build Guide

These instructions cover the chapters that now build both Blackwell (`sm_100`)
and Grace-Blackwell (`sm_121`) binaries via the shared `common/cuda_arch.mk`
configuration.

## Quick Reference

| Chapter | Build Command | Notes |
|---------|---------------|-------|
| ch1     | `make compare` | Produces `batched_gemm_example_sm100` / `_sm121` |
| ch2     | `make compare` | Builds NVLink + coherency demos for both CCs |
| ch4     | `make nvshmem` / `make compare` | NVSHMEM builds respect architecture suffixes |
| ch6     | `make compare` | All CUDA basics rebuilt for each arch |
| ch7     | `make compare` | Memory optimizations get `_sm100` / `_sm121` outputs |
| ch8     | `make compare` | Occupancy/control-flow samples per arch |
| ch9     | `make compare` | Fusion & CUTLASS demos linked for each CC |
| ch10    | `make compare` | Pipeline/TMA kernels built for both; Blackwell-only targets skip on GB10 |
| ch11    | `make compare` | Stream overlap suite with benchmark helpers |
| ch12    | `make compare` | CUDA Graphs + dynamic parallelism dual builds |

### Per-Architecture Targets

Every updated Makefile exposes:

- `make blackwell` – cleans and builds with `ARCH=sm_100`
- `make grace-blackwell` – cleans and builds with `ARCH=sm_121`

Binary names default to `_sm100` / `_sm121` suffixes. Set `USE_ARCH_SUFFIX=0`
before including `common/cuda_arch.mk` to retain legacy naming.

## CI Coverage

The `scripts/ci/run_compare_builds.sh` helper runs `make compare` for the
chapters listed above. The GitHub Actions workflow
`.github/workflows/dual-arch-compare.yml` invokes that script inside an NVIDIA
CUDA 13.0 container to confirm both codepaths stay buildable.

## Troubleshooting

- Ensure `nvcc` from CUDA Toolkit 13.0+ is on `PATH` when invoking the Makefiles.
- Chapter 4 can operate without NVSHMEM installed; builds fall back to an
  educational mode with stubbed symbols.
- For CI-free validation, run `./scripts/ci/run_compare_builds.sh` locally from
  the repo root; the script exits on the first failed build.
