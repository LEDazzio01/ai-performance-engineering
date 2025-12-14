# Chapter 3 - System Tuning

## Summary
Captures the host-level changes-NUMA pinning, governor tweaks, container settings, and Kubernetes manifests-that keep GPU workloads fed before kernel-level optimization begins.

## Learning Goals
- Diagnose CPU and memory affinity issues that throttle GPU pipelines.
- Harden Docker and Kubernetes environments for sustained GPU throughput on shared clusters.
- Automate repeatable system tuning via shell scripts so lab machines stay consistent.
- Quantify how host-level fixes raise GEMM throughput and reduce launch latency.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_numa_unaware.py`, `optimized_numa_unaware.py`, `bind_numa_affinity.py`, `numa_topology_script.sh` | NUMA diagnostics and pinning helpers for binding data loaders, NCCL ranks, and GPU contexts to the right CPU sockets. |
| `baseline_docker.py`, `optimized_docker.py`, `docker_gpu_optimized.dockerfile`, `system_tuning.sh`, `gpu_setup_commands.sh` | Container configs plus host setup scripts that toggle persistence mode, huge pages, IRQ steering, and MIG visibility. |
| `baseline_kubernetes.py`, `optimized_kubernetes.py`, `kubernetes_mig_pod.yaml`, `kubernetes_topology_pod.yaml` | Kubernetes manifests demonstrating topology-aware scheduling and MIG partitioning for multi-tenant fleets. |
| `cpu_gpu_numa_optimizations.sh`, `system_tuning.sh`, `gpu_setup_commands.sh` | Workflow scripts for aligning CPU governors, cgroup limits, persistence mode, and driver settings with the benchmark harness. |
| `baseline_gemm.py`, `optimized_gemm.py`, `train.py` | Simple GEMM + training loops that surface the impact of system tuning changes in measurable FLOP/s. |
| `compare.py`, `requirements.txt`, `expectations_b200.json` | Harness entry, Python deps, and regression thresholds. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python ch03/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch03
python -m cli.aisp bench run --targets ch03 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_b200.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- Run `python baseline_numa_unaware.py --diagnostics` before and after `bind_numa_affinity.py` to ensure cross-socket memory traffic drops to near zero.
- `python optimized_docker.py --image docker_gpu_optimized.dockerfile` should sustain the same throughput as host runs while keeping GPU clocks pinned.
- `python compare.py --examples gemm` shows optimized_gemm matching the measured host peak after applying `system_tuning.sh`.

## Notes
- `cpu_gpu_numa_optimizations.sh` is safe to rerun after every reboot; it re-applies irqbalance pinning and governor settings.
- Kubernetes manifests document the necessary annotations for NVLink/NVSwitch affinity without pointing to external repos.
