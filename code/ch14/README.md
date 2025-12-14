# Chapter 14 - Compiler & Triton Optimization

## Summary
Highlights compiler-driven acceleration: `torch.compile` workflows, Triton kernels, CUTLASS/TMA experimentation, and quantization-aware communication, all validated through the shared harness.

## Learning Goals
- Adopt `torch.compile` modes for large models while tracking compile-time and steady-state gains.
- Author Triton kernels (including TMA schedules) that rival custom CUDA.
- Profile FlexAttention and regional compilation strategies end-to-end.
- Blend quantization with NCCL and pipeline overlap without regressions.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_model_eager.py`, `optimized_model_eager.py`, `torch_compile_large_model.py`, `torch_compiler_examples.py`, `training_large_model_1_5x.py` | Model-scale examples showcasing compile modes, guard rails, and large-model sanity tests. |
| `baseline_cutlass.py`, `optimized_cutlass.py`, `triton_examples.py`, `triton_tma_blackwell.py`, `triton_fp8_advanced.py`, `triton_nvshmem_example.py` | CUTLASS vs Triton comparisons plus advanced TMA/NVSHMEM Triton kernels. |
| `baseline_flex_attention.py`, `optimized_flex_attention.py`, `test_flex_attention.py` | FlexAttention workloads that validate custom score mods, masks, and compile speedups. |
| `baseline_nccl_quantization.py`, `optimized_nccl_quantization.py`, `deepseek_innovation_l2_bypass.py` | Quantization-aware communication and the DeepSeek-inspired L2 bypass experiment. |
| `baseline_regional_triton.py`, `optimized_regional_triton.py`, `inspect_compiled_code.py`, `benchmark_tma_configs.py` | Regional compilation and TMA parameter sweeps for auto-tuning generated kernels. |
| `compare.py`, `requirements.txt`, `expectations_b200.json`, `train.py`, `transformer.py` | Harness entry plus model definitions and dependency pins. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python ch14/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch14
python -m cli.aisp bench run --targets ch14 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_b200.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python optimized_model_eager.py --profile minimal` produces compile-time summaries followed by steady-state throughput gains vs the baseline.
- `python triton_tma_blackwell.py --validate` compares Triton and CUDA outputs to double-check TMA scheduling logic.
- `python compare.py --examples flex_attention` shows the compiled path significantly reducing kernel launch count without changing accuracy.

## Notes
- `inspect_compiled_code.py` dumps Triton/PTX/Graph captures for any target; edit the helper to introspect new workloads.
- `requirements.txt` includes nightly Triton + PyTorch wheels to keep compiler features aligned with the CUDA 13 toolchain.
