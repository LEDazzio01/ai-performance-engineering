# Chapter 12 - CUDA Graphs & Dynamic Workloads

## Summary
Covers modern CUDA Graph capabilities-conditional capture, graph memory tuning, dynamic parallelism, and work queues-to keep irregular workloads performant without per-launch overhead.

## Learning Goals
- Capture steady-state workloads into CUDA Graphs and study the delta vs eager launches.
- Use conditional nodes and graph memory pools for adaptive pipelines.
- Experiment with device-side launches (dynamic parallelism) to reduce CPU involvement.
- Implement GPU-resident work queues and uneven partition schedulers.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_cuda_graphs.py`, `optimized_cuda_graphs.py`, `baseline_cuda_graphs_conditional*.cu`, `optimized_cuda_graphs_conditional*.cu` | Graph capture demos that evolve from simple replay to conditional and DSM-aware execution. |
| `baseline_graph_bandwidth.{py,cu}`, `optimized_graph_bandwidth.{py,cu}`, `baseline_kernel_launches.py`, `optimized_kernel_launches.py` | Launch- and bandwidth-focused studies illustrating how graphs reduce CPU overhead. |
| `baseline_dynamic_parallelism_host.cu`, `baseline_dynamic_parallelism_device.cu`, `optimized_dynamic_parallelism_host.cu`, `optimized_dynamic_parallelism_device.cu`, `dynamic_parallelism_sm121/` | Device-side launch samples showing when dynamic parallelism helps or hurts. |
| `baseline_work_queue.{py,cu}`, `optimized_work_queue.{py,cu}`, `work_queue_common.cuh` | GPU work queues for irregular batch sizes, including NVTX instrumentation. |
| `baseline_uneven_partition.cu`, `optimized_uneven_partition.cu`, `baseline_uneven_static.cu`, `optimized_uneven_static.cu` | Uneven workload partitioners that rebalance CTA assignments at runtime. |
| `baseline_kernel_fusion.{py,cu}`, `optimized_kernel_fusion.{py,cu}` | Kernel fusion exercises within graph capture so you can remove CPU synchronization entirely. |
| `compare.py`, `cuda_extensions/`, `expectations_gb10.json` | Harness entry, extension stubs, and expectation thresholds. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ch12
python compare.py --profile none
python tools/cli/aisp bench list-targets --chapter ch12
python tools/cli/aisp bench run --targets ch12 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_gb10.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python optimized_cuda_graphs.py --iterations 100` should report lower wall-clock time than the baseline while matching outputs.
- Device-side dynamic parallelism samples emit warnings on unsupported hardware, ensuring you only trust data from GPUs with the feature enabled.
- `python optimized_work_queue.py --trace` exposes balanced dequeue times across CTAs when compared to the baseline's stragglers.

## Notes
- `cuda_graphs_workload.cuh` holds reusable graph capture helpers when you want to wrap your own kernels.
- `helper_*.cu` files contain host/device glue for the dynamic-parallelism case studies-copy them when bootstrapping new experiments.
