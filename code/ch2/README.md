# Chapter 2 - GPU Hardware Architecture

## Summary
Provides architecture awareness tooling for Blackwell-era systems-query SM and memory specs, validate NVLink throughput, and experiment with CPU-GPU coherency so optimizations stay grounded in measured hardware limits.

## Learning Goals
- Query and log GPU, CPU, and fabric capabilities before running performance studies.
- Measure NVLink, PCIe, and memory-bandwidth ceilings using purpose-built microbenchmarks.
- Validate Grace-Blackwell coherency paths to know when zero-copy buffers help or hurt.
- Contrast baseline vs optimized cuBLAS invocations to highlight architecture-specific tuning levers.

## Directory Layout
| Path | Description |
| --- | --- |
| `hardware_info.py`, `cpu_gpu_topology_aware.py` | System scanners that record GPU capabilities, NUMA layout, NVLink/NVSwitch connectivity, and affinity hints. |
| `baseline_uma_memory_reporting.py`, `optimized_uma_memory_reporting.py`, `uma_memory_utils.py` | UMA memory reporting pair: baseline trusts `cudaMemGetInfo`, optimized folds in `/proc/meminfo` (MemAvailable + reclaimable SwapFree) for DGX Spark/iGPU platforms where nvidia-smi hides framebuffer stats. |
| `nvlink_c2c_bandwidth_benchmark.py`, `baseline_memory_transfer.py`, `optimized_memory_transfer.py`, `optimized_memory_transfer_nvlink.cu`, `optimized_memory_transfer_zero_copy.cu` | Peer-to-peer and zero-copy experiments for quantifying NVLink, PCIe, and coherent memory performance. |
| `cpu_gpu_grace_blackwell_coherency.cu`, `cpu_gpu_grace_blackwell_coherency_sm121` | Grace-Blackwell cache-coherent samples that compare explicit transfers vs shared mappings. |
| `baseline_cublas.py`, `optimized_cublas.py` | cuBLAS GEMM benchmark pair that toggles TF32, tensor op math, and stream affinity to highlight architecture knobs. |
| `compare.py`, `Makefile`, `expectations_gb10.json` | Harness driver, CUDA build rules, and expectation file for automated pass/fail checks. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ch2
python compare.py --profile none
python tools/cli/aisp bench list-targets --chapter ch2
python tools/cli/aisp bench run --targets ch2 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_gb10.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python hardware_info.py --json artifacts/hw_snapshot.json` records the correct device name, SM count, and HBM size for every GPU in the system.
- `python nvlink_c2c_bandwidth_benchmark.py --gpus 0 1` sustains ~250 GB/s unidirectional on NVLink-connected pairs; mismatches flag topology or driver issues.
- Running the coherency sample shows zero-copy benefiting sub-MB transfers while large transfers favor explicit H2D copies, matching the documented thresholds.

## Notes
- Grace-only coherency tests require GB200/GB300 nodes; the binaries no-op on PCIe-only hosts.
- `Makefile` builds both CUDA and CPU tools so results can be compared without leaving the chapter.
- When nvidia-smi shows `Memory-Usage: Not Supported` on UMA/iGPU boxes, run `python optimized_uma_memory_reporting.py` to see a MemAvailable + SwapFree view (baseline shows raw `cudaMemGetInfo` for comparison). Add `--snapshot` to persist a JSON under `artifacts/uma_memory_snapshots/` for historical tracking.
