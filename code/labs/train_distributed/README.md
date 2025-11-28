# Lab - Distributed Training Playbook

## Summary
Collects distributed-training recipes for Blackwell clusters: DDP, FSDP, FSDP2+FP8, ZeRO-1/2/3, symmetric memory, and flash-attention-aware all-reduce handling, all runnable through the harness.

## Standalone vs HuggingFace Scripts

| Script Type | Files | Dependencies | Use Case |
|-------------|-------|--------------|----------|
| **Standalone** | `*_standalone.py` | PyTorch only | Quick testing, no downloads |
| **HuggingFace** | `baseline_fsdp.py`, `optimized_fsdp.py` | transformers, TinyStories | Production-like benchmarks |

> **Quick Start**: Use `baseline_fsdp2_standalone.py` and `optimized_fsdp2_fp8_standalone.py` for immediate
> FSDP2+FP8 benchmarks without HuggingFace model downloads.

## Learning Goals
- Benchmark standard DDP vs optimized overlap-aware variants.
- Exercise FSDP and ZeRO strategies with shared helper utilities.
- Validate symmetric-memory training modes that pool NVLink bandwidth.
- Reuse launcher utilities (torchrun) with consistent configuration.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_ddp.py`, `optimized_ddp.py`, `baseline_ddp_flash.py`, `optimized_ddp_flash.py`, `ddp.py` | DDP workloads including flash-attention aware overlap tuning. |
| `baseline_fsdp.py`, `optimized_fsdp.py`, `train_fsdp.py` | Native-torchrun FSDP2 scripts (BF16 baseline + FP8 optimized) with built-in throughput/memory instrumentation. |
| `baseline_symmem_training.py`, `optimized_symmem_training.py` | Symmetric-memory strategies for optimizer state replication. |
| `baseline_zero1.py`, `baseline_zero2.py`, `baseline_zero3.py`, `optimized_zero1.py`, `optimized_zero2.py`, `optimized_zero3.py`, `zero1.py`, `zero2.py`, `zero3.py` | ZeRO implementations (1/2/3) plus helpers for parameter partitioning. |
| `pipeline.py`, `pipeline_gpipe.py`, `pipeline_1f1b.py`, `pipeline_dualpipe.py`, `pipeline_dualpipev.py`, `baseline_pipeline_*.py`, `optimized_pipeline_*.py` | Toy pipeline-parallel executors (GPipe / 1F1B / DualPipe / DualPipeV) with telemetry and harness integration. |
| `training_utils/`, `utils.py`, `__init__.py` | Shared launch utilities, argument parsing, and harness exports. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ai-performance-engineering
python -m cli.aisp bench list-targets --chapter labs/train_distributed
python -m cli.aisp bench run --targets labs/train_distributed --profile minimal
```
- Targets follow the `labs/train_distributed:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/train_distributed:<workload>="--flag value"` to sweep schedule knobs.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/train_distributed --profile minimal` runs every distributed configuration registered with the harness.
- `torchrun --standalone --nproc_per_node 1 labs/train_distributed/train_fsdp.py --mode baseline --steps 8 --sequence-length 2048` exercises the BF16 baseline; swap `--mode optimized` to enable the FP8 path (requires `torchao`). Both variants print steps/s, tokens/s, TFLOPs per rank, and peak GPU memory.
- `python labs/train_distributed/optimized_zero3.py --summary` shows reduced peak memory vs the baseline script.

## FSDP2 + FP8 Walkthrough
- `train_fsdp.py` dispatches between `baseline_fsdp.py` (BF16 FSDP2 sharding) and `optimized_fsdp.py` (torchao FP8 + grad checkpoint + fused AdamW). No Accelerate dependency remains—the scripts initialize NCCL, wrap Hugging Face models with FSDP2, and rely on the shared TinyStories loader.
- Baseline defaults: per-rank microbatch size 1, BF16 mixed precision, Forward/Backward prefetch enabled. Use `--sequence-length`, `--micro-batch-size`, `--grad-accum`, and `--steps` to sweep without editing env vars.
- Optimized defaults: per-rank microbatch size 2, gradient accumulation 2, torchao `Float8LinearConfig(enable_fsdp_float8_all_gather=True)`, gradient clipping, and persistent dataloader workers. The script aborts with a helpful error if `torchao` is missing.
- Instrumentation: both variants use `ThroughputTracker` + `gpu_memory_usage()` to log steps/s, tokens/s, TFLOPs per rank, and CUDA peak memory (active/reserved) each time an optimizer step completes. Logs are emitted only from rank 0, making them easy to scrape in the harness.
- Example launch (single GPU, short sanity run):
  ```bash
  torchrun --standalone --nproc_per_node 1 labs/train_distributed/train_fsdp.py \
    --mode optimized --steps 4 --sequence-length 1024 --micro-batch-size 1 --grad-accum 2
  ```

## Pipeline Parallelism Demos
- Targets: `labs/train_distributed:pipeline_gpipe_2stages`, `labs/train_distributed:1f1b_2stages`, `labs/train_distributed:dualpipe_2stages`, and the V-shape variant `labs/train_distributed:dualpipev_2stages`. Each exposes `baseline_` vs `optimized_` scripts so you can quantify idle bubble reductions with just two GPUs.
- Harness launch examples:
  ```bash
  python -m cli.aisp bench run --targets labs/train_distributed:pipeline_gpipe_2stages --profile minimal
  python -m cli.aisp bench run --targets labs/train_distributed:1f1b_2stages --profile minimal
  python -m cli.aisp bench run --targets labs/train_distributed:dualpipe_2stages --profile minimal
  python -m cli.aisp bench run --targets labs/train_distributed:dualpipev_2stages --profile minimal
  ```
- Direct runs (single process controlling four GPUs) let you sweep microbatch knobs and capture telemetry:
  ```bash
  python labs/train_distributed/pipeline_gpipe.py --mode baseline --steps 4 --n-stages 2
  python labs/train_distributed/pipeline_gpipe.py --mode optimized --steps 4 --micro-batch-size 8 --n-stages 2
  python labs/train_distributed/pipeline_1f1b.py --mode optimized --steps 6 --micro-batch-target 8 --n-stages 2
  python labs/train_distributed/pipeline_dualpipe.py --mode optimized --steps 8 --dual-window 16 --n-stages 2
  python labs/train_distributed/pipeline_dualpipev.py --mode optimized --steps 8 --dual-window 16 --n-stages 2
  ```
- Every run prints stage utilization plus max forward/backward queue depth so you can mirror the bubble diagrams from Nsight Systems. The optimized variants default to smaller microbatches, non-blocking transfers, and deeper dual windows (for DualPipe/DualPipeV) to prove the utilization gains relative to the baselines. Use `--n-stages` to change how many GPUs are required; the toy stage splitting lives in `pipeline.py`.

## Notes
- Set `TORCHRUN_ARGS` or pass `--torchrun-env` via the CLI when launching multi-node tests.
- `utils.py` exposes helper functions (like `resolve_topology()`) that can be reused in other labs.
