# Lab - FlashAttention Gluon

## Summary
Contrasts a naive unfused attention (matmul + softmax + matmul) with a fused, warp-specialized FlashAttention kernel. The optimized path prefers a Gluon/Triton kernel; if Gluon is unavailable, it falls back to the flash-attn fused kernel (warp-specialized on Blackwell).

## Learning Goals
- Quantify the benefit of fused FlashAttention vs unfused math softmax.
- Observe warp specialization, TMA loads, and softmax tiling in NVTX traces.
- Verify Gluon-first, flash-attn fallback provider selection on Blackwell hardware.
- Capture harness artifacts to track regressions across providers.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_flashattention_gluon.py` | Unfused attention with math softmax path. |
| `optimized_flashattention_gluon.py` | Fused Gluon/flash-attn kernel with warp specialization and provider fallback. |
| `flashattention_gluon_common.py`, `__init__.py` | Shared helpers and harness target exports. |

## Running the Benchmarks
```bash
cd ai-performance-engineering
python -m cli.aisp bench list-targets --chapter labs/flashattention_gluon
python -m cli.aisp bench run --targets labs/flashattention_gluon:baseline_flashattention_gluon --profile minimal
python -m cli.aisp bench run --targets labs/flashattention_gluon:optimized_flashattention_gluon --profile minimal
```
- Set `--profile none` for fast correctness checks or keep `--profile minimal` for Nsight-ready traces.

## Validation Checklist
- NVTX ranges appear as `flashattention_baseline_unfused` vs `flashattention_optimized_<provider>`.
- Provider metric reports `gluon` when available, otherwise `flash-attn`; runs do not fail when Gluon is missing.
- Optimized path shows fewer kernels and higher throughput than the baseline in harness output.
- Expectations stay green on Blackwell; update thresholds if new hardware changes performance envelopes.

## Notes
- Requires CUDA-capable GPU plus Gluon or flash-attn (installed via `setup.sh` when possible).
- Use the checklist when porting a naive FlashAttention to a warp-specialized, TMA-driven kernel on Blackwell: explicit block/swizzle layouts, softmax subtiles, ordered M-barriers, tensor memory placement, TMA bulk loads, and masking folded into softmax partitions.
