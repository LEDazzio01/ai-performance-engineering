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
| `baseline_rack_prep.py`, `optimized_rack_prep.py`, `grace_blackwell_topology.py` | Rack-prep pair: baseline uses pageable staging and leaves IRQ/RPS/XPS unpinned; optimized binds workers to NIC-local CPUs, uses pinned double-buffered staging, and emits the IRQ/RPS/XPS plan for the selected NICs. |
| `baseline_docker.py`, `optimized_docker.py`, `docker_gpu_optimized.dockerfile`, `system_tuning.sh`, `gpu_setup_commands.sh` | Container configs plus host setup scripts that toggle persistence mode, huge pages, IRQ steering, and MIG visibility. |
| `baseline_kubernetes.py`, `optimized_kubernetes.py`, `kubernetes_mig_pod.yaml`, `kubernetes_topology_pod.yaml` | Kubernetes manifests demonstrating topology-aware scheduling and MIG partitioning for multi-tenant fleets. |
| `cpu_gpu_numa_optimizations.sh`, `system_tuning.sh`, `gpu_setup_commands.sh` | Workflow scripts for aligning CPU governors, cgroup limits, persistence mode, and driver settings with the benchmark harness. |
| `baseline_gemm.py`, `optimized_gemm.py`, `train.py` | Simple GEMM + training loops that surface the impact of system tuning changes in measurable FLOP/s. |
| `compare.py`, `requirements.txt`, `expectations_gb10.json` | Harness entry, Python deps, and regression thresholds. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ch3
python compare.py --profile none
python tools/cli/aisp bench list-targets --chapter ch3
python tools/cli/aisp bench run --targets ch3 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_gb10.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python compare.py --examples rack_prep` shows optimized runs pinning NUMA/IRQ/RPS/XPS correctly while beating the baseline on GEMM throughput.
- `python optimized_rack_prep.py --apply --nic <if0> [--nic <if1>]` writes affinity settings and reports pinned queues/processors without errors.
- `python optimized_docker.py --validate` emits container/runtime settings that match the host tuning output.
- GEMM/ILP helpers under `train.py` and `baseline_gemm.py` vs `optimized_gemm.py` produce higher FLOP/s once the host tuning is applied.

## Notes
- Rack prep quickstart (NIC/GPU/NUMA): select NIC order with repeated `--nic`, run the baseline to capture the pageable, topology-unaware reference, then run `optimized_rack_prep.py` (dry-run by default) to print IRQ/RPS/XPS changes. Use `--apply` for host writes and validate via the printed affinity report.
- Via `aisp bench`, pass per-target flags with `--target-extra-arg` to tune NICs and reserved CPUs: `python tools/cli/aisp bench run --targets ch3:rack_prep --target-extra-arg 'ch3:rack_prep=--apply --reserve 2 --nic enp193s0f0np0'`.
- System scripts assume sudo when applying host settings; keep them dry-run on shared/dev hosts.

## Validation Checklist
- Run `python baseline_numa_unaware.py --diagnostics` before and after `bind_numa_affinity.py` to ensure cross-socket memory traffic drops to near zero.
- `python optimized_docker.py --image docker_gpu_optimized.dockerfile` should sustain the same throughput as host runs while keeping GPU clocks pinned.
- `python compare.py --examples gemm` shows optimized_gemm matching the measured host peak after applying `system_tuning.sh`.

## Notes
- `cpu_gpu_numa_optimizations.sh` is safe to rerun after every reboot; it re-applies irqbalance pinning and governor settings.
- Kubernetes manifests document the necessary annotations for NVLink/NVSwitch affinity without pointing to external repos.
