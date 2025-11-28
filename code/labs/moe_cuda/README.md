# Lab - CUDA MoE Decode Toolkit

## Summary
Implements mixture-of-experts decode helpers directly in CUDA: decode kernels, KV-transfer graphs, router policies, and validation math so you can iterate on Blackwell-friendly pipelines.

## Learning Goals
- Benchmark decode kernels that stage tokens through shared memory and cp.async pipelines.
- Optimize KV-transfer strategies (manual, CUDA Graphs) across NVLink fabrics.
- Prototype routers that understand MoE grouping, locality, and vectorized loads.
- Validate CUDA kernels against Python math models before integrating into serving stacks.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_decode_attention.py`, `optimized_decode_attention.py`, `optimized_decode_attention_math.py` | Attention microbenchmarks plus math-only checks to vet numerical stability. |
| `baseline_decode_kernel.py`, `optimized_decode_kernel.py`, `decode_kernels.cu`, `kernels/` | CUDA kernels and wrappers for the decode core. |
| `baseline_kv_transfer.py`, `optimized_kv_transfer.py`, `optimized_kv_transfer_graphs.py` | KV-transfer samples comparing eager vs CUDA Graph orchestration. |
| `baseline_router.py`, `optimized_router.py`, `optimized_router_vectorized.py` | MoE router logic fit for device execution. |
| `expectations_gb10.json`, `__init__.py` | Metadata and module exports needed by the harness. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ai-performance-engineering
python -m cli.aisp bench list-targets --chapter labs/moe_cuda
python -m cli.aisp bench run --targets labs/moe_cuda --profile minimal
```
- Targets follow the `labs/moe_cuda:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/moe_cuda:<workload>="--flag value"` to sweep schedule knobs.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/moe_cuda --profile minimal` runs every baseline/optimized pair and captures NVTX traces.
- `python labs/moe_cuda/optimized_decode_attention_math.py --validate` compares the CUDA path to the math reference and fails loudly if drift is detected.
- KV transfer graphs print latency breakdowns showing overlap improvements relative to the baseline script.

## Notes
- `kernels/` houses the raw CUDA sources split by component; edit schedules there before rebuilding via the harness.
- `optimized_kv_transfer_graphs.py` emits CUDA Graph captures under `artifacts/` for reproducibility.
