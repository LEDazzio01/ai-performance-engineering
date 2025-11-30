# Performance Intake & Triage Bundle

This repo already collects benchmark artifacts, but two quick additions make it easier to align on goals and capture proof-of-benefit for new workloads.

## Quick Actions
- Fill the one-page intake (`templates/performance_intake.yaml`) so goals, SLOs, workload shape, and constraints are explicit.
- Run the triage bundle to gather a clean baseline: `core/scripts/profiling/perf_triage_bundle.sh --output-root ./artifacts --tag smoke -- <your command>`.
- Pick 2–3 high-ROI experiments (below) and A/B them with the same inputs, logging before/after metrics.

## One-Page Intake
Copy `templates/performance_intake.yaml` and fill it for the workload under test. The fields cover KPIs, workload shape, SLOs, hardware/software topology, current baseline, and guardrails—enough to reason about goodput and cost levers without guesswork.

## 30-Minute Triage Bundle
The bundle captures hardware/software facts plus a short run with either Nsight Systems (when present) or `nvidia-smi dmon` as a fallback.

Usage:
```bash
# Snapshots only (no runtime command)
core/scripts/profiling/perf_triage_bundle.sh --output-root ./artifacts

# Snapshots + runtime capture for a representative command
core/scripts/profiling/perf_triage_bundle.sh --output-root ./artifacts --tag smoke --nsys -- \
  python ch01/baseline_matmul.py --batch-size 32
```

What it does:
- Creates `artifacts/perf_triage_<host>_<timestamp>[_<tag>]` with GPU/CPU/memory/storage/network snapshots, CUDA/PyTorch versions, and manifest metadata.
- If Nsight Systems is available (and not disabled), runs the provided command under `nsys profile -t cuda,nvtx,osrt,cudnn,cublas` and emits both the `.nsys-rep` and a text summary.
- Otherwise, samples `nvidia-smi dmon` while the command runs and stores a CSV timeseries for SM%, mem BW, power, and utilization.
- Packs everything into a `.tgz` for sharing.

### PyTorch Compiler Diagnostics (optional)
If you use `torch.compile`, enable graph-break diagnostics before the run:
```bash
export TORCH_LOGS="+dynamo,+inductor,perf_hints,output_code"
export TORCH_COMPILE_DEBUG=1
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export TORCHINDUCTOR_BENCHMARK_KERNEL=1
```
Inside Python you can inspect breaks with:
```python
torch._dynamo.explain(model)(*example_inputs)
```

## High-ROI Experiment Menu (run A/B with identical inputs)
- **Inference:** continuous batching and scheduling, speculative decoding (EAGLE/Medusa), quantization (FP8/INT8/INT4 with quality gates), grammar/constraint overhead checks, topology-aware placement (NVLink/NVSwitch first), and engine comparisons (vLLM/SGLang/TensorRT-LLM).
- **Training:** dataloader throughput (prefetch, pinned memory, CPU workers), `torch.compile` with graph-break fixes or regional compilation, overlap of compute/communication (bucket sizing + NCCL env), parallelism mix (DP/FSDP/TP/PP/MoE balance), mixed precision + fused ops, and small autotuning sweeps for batch/seq length/kernel configs.
- **Both:** measure goodput (useful GPU work / wall time), track throughput/$ and throughput/W across instance types, and keep lightweight continuous monitors (DCGM/Prometheus) to catch regressions.

## What to Return
- Completed intake YAML.
- The triage bundle `.tgz` containing system snapshots and Nsight/dmon outputs.
- A/B table with throughput, p50/p99 latency, GPU SM%, NIC GB/s, and any accuracy deltas.
