# Chapter 13 - PyTorch Profiling & Memory Tuning

## Summary
Focuses on PyTorch-centric optimizations: compiled autograd, memory profiling, FSDP/context/expert parallelism, and FP8/quantization workflows backed by the same harness infrastructure.

## Learning Goals
- Profile PyTorch training loops end-to-end, capturing goodput, memory, and kernel traces.
- Apply `torch.compile`, regional compilation, and custom allocators to reduce overhead.
- Tune DataLoader, KV-cache, and optimizer states to eliminate fragmentation.
- Exercise FP8/quantized training recipes with Transformer Engine integration.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_training_standard.py`, `optimized_training_standard.py`, `train.py`, `train_deepseek_v3.py`, `train_deepseek_coder.py` | Reference training loops showcasing eager vs compiled paths and DeepSeek-inspired configs. |
| `baseline_dataloader_default.py`, `optimized_dataloader_default.py`, `baseline_memory_profiling.py`, `optimized_memory_profiling.py`, `memory_profiling.py` | DataLoader/memory studies that explain how to read allocator stats and fix leaks. |
| `baseline_attention_standard.py`, `optimized_attention_standard.py`, `baseline_arithmetic_intensity.py`, `optimized_arithmetic_intensity.py`, `baseline_matmul_pytorch.py`, `optimized_matmul_pytorch.py` | Attention and matmul microbenchmarks tuned purely within PyTorch. |
| `baseline_context_parallelism.py`, `optimized_context_parallelism.py`, `baseline_expert_parallelism.py`, `optimized_expert_parallelism.py`, `fsdp_example.py` | Context, expert, and FSDP sharding demos for scaling beyond a single GPU. |
| `baseline_precisionfp8*.py`, `optimized_precisionfp8*.py`, `baseline_precisionmixed.py`, `optimized_precisionmixed.py`, `compiled_autograd.py` | Precision-management suites covering Transformer Engine and compiled autograd recipes. |
| `baseline_quantization.py`, `optimized_quantization.py`, `baseline_kv_cache_naive.py`, `optimized_kv_cache.py` | Quantization and KV-cache pipelines for inference/training memory savings. |
| `compare.py`, `compare_perf.py`, `requirements.txt`, `expectations_gb10.json`, `workload_config.py` | Harness entry, performance comparison helper, dependencies, and regression baselines. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ch13
python compare.py --profile none
python tools/cli/aisp bench list-targets --chapter ch13
python tools/cli/aisp bench run --targets ch13 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_gb10.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python compare.py --examples training_standard` shows optimized training runs producing higher goodput with identical metrics.
- `python optimized_precisionfp8_te.py --validate` confirms Transformer Engine calibration plus NVFP8 execution with max error tolerances enforced.
- `python memory_profiling.py --dump` and the optimized variant demonstrate allocator fragmentation dropping after applying the recommended knobs.
- `python optimized_context_parallelism.py --sequence-length 131072 --cp-ranks 4` shreds a long context across four GPUs; adjust `--sequence-length` / `--cp-ranks` to mirror TorchTitan-style CP tuning.

## Notes
- `custom_allocator.py` contains a standalone torch allocator shim that can be re-used in other chapters when debugging fragmentation.
- `compiled_autograd.py` doubles as a tutorial on partial graph capture; the README here references it directly.
