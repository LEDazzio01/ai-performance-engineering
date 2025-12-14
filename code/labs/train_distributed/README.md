# Lab - Distributed Training Playbook

## Summary
Collects distributed-training recipes for Blackwell clusters: DDP, FSDP, ZeRO-1/2/3, symmetric memory, and flash-attention-aware all-reduce handling, all runnable through the harness.

## Learning Goals
- Benchmark standard DDP vs optimized overlap-aware variants.
- Exercise FSDP and ZeRO strategies with shared helper utilities.
- Validate symmetric-memory training modes that pool NVLink bandwidth.
- Reuse launcher utilities (torchrun) with consistent configuration.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_ddp.py`, `optimized_ddp.py`, `baseline_ddp_flash.py`, `optimized_ddp_flash.py`, `ddp.py` | DDP workloads including flash-attention aware overlap tuning. |
| `baseline_fsdp.py`, `optimized_fsdp.py`, `train_fsdp.py` | FSDP scripts that demonstrate shard-by-shard memory savings. |
| `baseline_symmem_training.py`, `optimized_symmem_training.py` | Symmetric-memory strategies for optimizer state replication. |
| `baseline_zero1.py`, `baseline_zero2.py`, `baseline_zero3.py`, `optimized_zero1.py`, `optimized_zero2.py`, `optimized_zero3.py`, `zero1.py`, `zero2.py`, `zero3.py` | ZeRO implementations (1/2/3) plus helpers for parameter partitioning. |
| `training_utils/`, `utils.py`, `__init__.py` | Shared launch utilities, argument parsing, and harness exports. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/train_distributed
python -m cli.aisp bench run --targets labs/train_distributed --profile minimal
```
- Targets follow the `labs/train_distributed:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/train_distributed:<workload>="--flag value"` to sweep schedule knobs.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/train_distributed --profile minimal` runs every distributed configuration registered with the harness.
- `python labs/train_distributed/train_fsdp.py --validate` confirms numerical parity between FSDP shards and the baseline DDP path.
- `python labs/train_distributed/optimized_zero3.py --summary` shows reduced peak memory vs the baseline script.

## Notes
- Set `TORCHRUN_ARGS` or pass `--torchrun-env` via the CLI when launching multi-node tests.
- `utils.py` exposes helper functions (like `resolve_topology()`) that can be reused in other labs.
