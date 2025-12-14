# Lab - FlexAttention Harness

## Summary
Mirrors the FlexAttention CuTe DSL walkthrough: run eager vs compiled FlexAttention, compare to the CuTe path, and experiment with block masks, score modifiers, and Triton-style compilation.

## Learning Goals
- Benchmark FlexAttention eager mode against compiled variants using identical masks/score mods.
- Validate CuTe-based FlashAttention fallbacks for platforms where FlexAttention is not available.
- Sweep sparsity knobs (block size, doc span) without editing source.
- Collect Nsight traces showing kernel fusion improvements after compiling.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_flex_attention.py`, `optimized_flex_attention.py` | FlexAttention DSL workloads toggling `torch.compile` for fused kernels. |
| `flex_attention_cute.py` | CuTe/FlashAttention tool for hardware without FlexAttention bindings. |
| `flexattention_common.py`, `expectations_b200.json` | Shared input builders, score modifiers, and regression thresholds. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/flexattention
python -m cli.aisp bench run --targets labs/flexattention --profile minimal
```
- Targets follow the `labs/flexattention:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/flexattention:<workload>="--flag value"` to sweep schedule knobs.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/flexattention:flex_attention --profile minimal` captures the eager vs compiled delta and stores artifacts.
- `BLOCK_SIZE=64 DOC_SPAN=128 python -m cli.aisp bench run --targets labs/flexattention:flex_attention` demonstrates masked sparsity sweeps.
- `python -m cli.aisp tools flex-attention-cute -- --batch 2 --seq-len 1024` succeeds even on systems missing FlexAttention bindings.

## Notes
- Environment variables such as `BLOCK_SIZE`, `DOC_SPAN`, and `TORCH_COMPILE_MODE` are read at runtime for quick experiments.
- Artifacts include NVTX traces; feed them to `core/analysis/deep_profiling_report.py` for convenience.
