"""Generate consistent README files for chapters and labs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RunSection:
    """Command guidance for running benchmarks."""

    commands: Sequence[str]
    notes: Sequence[str] = field(default_factory=tuple)


@dataclass
class Entry:
    """README content definition."""

    title: str
    summary: str
    goals: Sequence[str]
    contents: Sequence[Tuple[str, str]]
    validation: Sequence[str]
    run: Optional[RunSection] = None
    notes: Sequence[str] = field(default_factory=tuple)


def _format_markdown(entry: Entry) -> str:
    """Create markdown using the shared layout."""
    lines: List[str] = []
    lines.append(f"# {entry.title}")
    lines.append("")
    lines.append("## Summary")
    lines.append(entry.summary.strip())
    lines.append("")
    lines.append("## Learning Goals")
    for goal in entry.goals:
        lines.append(f"- {goal}")
    lines.append("")
    lines.append("## Directory Layout")
    lines.append("| Path | Description |")
    lines.append("| --- | --- |")
    for path, desc in entry.contents:
        lines.append(f"| {path} | {desc} |")
    lines.append("")
    if entry.run:
        lines.append("## Running the Benchmarks")
        lines.append("Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.")
        lines.append("```bash")
        for cmd in entry.run.commands:
            lines.append(cmd)
        lines.append("```")
        for note in entry.run.notes:
            lines.append(f"- {note}")
        lines.append("")
    lines.append("## Validation Checklist")
    for item in entry.validation:
        lines.append(f"- {item}")
    if entry.notes:
        lines.append("")
        lines.append("## Notes")
        for note in entry.notes:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _chapter_run_commands(slug: str) -> RunSection:
    """Default run commands for a chapter directory."""
    commands = [
        f"cd {slug}",
        "python compare.py --profile none",
        f"python tools/cli/aisp bench list-targets --chapter {slug}",
        f"python tools/cli/aisp bench run --targets {slug} --profile minimal",
    ]
    notes = [
        "Override `--profile` or `--iterations` per workload when capturing Nsight traces.",
        "Expectation baselines live next to each chapter in `expectations_gb10.json`; refresh with `--update-expectations` after validating new hardware.",
    ]
    return RunSection(commands=commands, notes=notes)


def _lab_run_commands(slug: str) -> RunSection:
    """Default run commands for a lab that is exposed through the CLI."""
    commands = [
        "cd ai-performance-engineering",
        f"python tools/cli/aisp bench list-targets --chapter {slug}",
        f"python tools/cli/aisp bench run --targets {slug} --profile minimal",
    ]
    notes = [
        f"Targets follow the `{slug}:<workload>` naming convention listed by `list-targets`.",
        f"Use `--target-extra-arg {slug}:<workload>=\"--flag value\"` to sweep schedule knobs.",
    ]
    return RunSection(commands=commands, notes=notes)


def chapter_entry(
    slug: str,
    title: str,
    summary: str,
    goals: Sequence[str],
    contents: Sequence[Tuple[str, str]],
    validation: Sequence[str],
    notes: Sequence[str] = (),
) -> Entry:
    """Create a chapter README entry."""
    return Entry(
        title=title,
        summary=summary,
        goals=goals,
        contents=contents,
        validation=validation,
        run=_chapter_run_commands(slug),
        notes=notes,
    )


def lab_entry(
    slug: str,
    title: str,
    summary: str,
    goals: Sequence[str],
    contents: Sequence[Tuple[str, str]],
    validation: Sequence[str],
    notes: Sequence[str] = (),
    run: Optional[RunSection] = None,
) -> Entry:
    """Create a lab README entry."""
    return Entry(
        title=title,
        summary=summary,
        goals=goals,
        contents=contents,
        validation=validation,
        run=run if run else _lab_run_commands(slug),
        notes=notes,
    )


ENTRIES: Dict[str, Entry] = {}

ENTRIES["README.md"] = Entry(
    title="AI Systems Performance Engineering",
    summary=dedent(
        """\
        Reference implementation of high-performance PyTorch, CUDA, and Triton workloads for NVIDIA Blackwell platforms.
        The repository packages 20 focused chapters, advanced labs, and the shared benchmarking harness so you can profile baselines, apply optimizations, and capture artifacts that prove performance gains."""
    ),
    goals=[
        "Understand how the chapters, labs, and shared tooling fit together.",
        "Stand up a reproducible environment for PyTorch 2.10-dev + CUDA 13 workloads on Blackwell GPUs.",
        "Run the benchmark harness directly or through the Typer CLI for automated artifact capture.",
        "Validate peak hardware characteristics before grading optimizations against stored expectations.",
    ],
    contents=[
        ("`ch1` - `ch20`", "One directory per chapter with baseline/optimized benchmarks, workload configs, and `compare.py` harness entrypoints."),
        ("`labs/`", "Deep-dive labs for matmul, routing, FlexAttention, MoE, persistent decode, distributed training, and more."),
        ("`common/python/`", "Shared benchmark harness, logging, workload metadata, and profiling utilities used by every chapter."),
        ("`tools/cli/aisp bench`", "Typer-based CLI for running, profiling, and verifying targets with reproducible artifacts."),
        ("`docs/` + `scripts/`", "Operational guides, profiling workflows, and setup/reset helpers (`setup.sh`, `cleanup.py`, `reset-gpu.sh`)."),
    ],
    run=RunSection(
        commands=[
            "cd ai-performance-engineering",
            "python3 -m venv .venv && source .venv/bin/activate",
            "pip install -r requirements_latest.txt",
            "python tools/cli/aisp bench list-targets --chapter ch1",
            "python tools/cli/aisp bench run --targets ch1 --profile minimal",
        ],
        notes=[
            "`setup.sh` installs system prerequisites (drivers, CUDA, Nsight) and should be rerun after driver upgrades.",
            "Use `python tools/testing/run_all_benchmarks.py --targets ch*` for automated regression suites.",
            "`python tools/analysis/analyze_expectations.py --artifacts-dir artifacts` compares new runs to stored thresholds.",
        ],
    ),
    validation=[
        "`pytest tests/integration` succeeds to confirm harness discovery and CLI plumbing.",
        "`python tools/benchmarking/benchmark_peak.py` reports TFLOP/s, bandwidth, and NVLink numbers close to the published ceilings.",
        "`python tools/cli/aisp bench verify --targets all` completes without regressions before refreshing expectation files.",
    ],
    notes=[
        "`tools/profiling/profile.sh` and `ncu_template.ini` capture Nsight traces with consistent metric sets.",
        "`benchmark_profiles/` and `artifacts/` hold run outputs; clean them via `python cleanup.py` when rotating hardware.",
        "`docs/perf_intake_and_triage.md` outlines the standard intake bundle for performance investigations.",
    ],
)

ENTRIES["ch1"] = chapter_entry(
    slug="ch1",
    title="Chapter 1 - Performance Fundamentals",
    summary=dedent(
        """\
        Establishes the baseline benchmarking discipline: measure goodput, remove Python-side stalls, and apply foundational CUDA optimizations such as pinned memory, batched transfers, and CUDA Graphs."""
    ),
    goals=[
        "Profile PyTorch training loops with the shared harness to identify memory-transfer and launch bottlenecks.",
        "Apply pinned memory, tensor preallocation, and larger batch sizes to raise GPU utilization.",
        "Capture and replay CUDA Graphs to trim kernel launch overhead for steady-state workloads.",
        "Compare hand-written GEMM kernels in batched vs. strided forms to understand arithmetic intensity.",
    ],
    contents=[
        ("`baseline_performance.py`, `optimized_performance.py`", "Goodput-focused training loop pair that toggles pinned memory, CUDA Graphs, and tensor preallocation."),
        ("`baseline_guided_decoding.py`, `optimized_guided_decoding.py`, `baseline_guided_decoding_math.py`, `optimized_guided_decoding_math.py`", "Guided decoding microbenchmark plus math-only validator for deterministic correctness checks."),
        ("`baseline_ilp_basic.py`, `optimized_ilp_basic.py`", "Instruction-level parallelism exercises that expose loop unrolling and register reuse benefits."),
        ("`baseline_warp_specialization.py`, `optimized_warp_specialization.py`", "Warp-specialized producer/consumer example that demonstrates overlapped memory movement."),
        ("`baseline_gemm.cu`, `optimized_gemm_batched.cu`, `optimized_gemm_strided.cu`", "CUDA GEMM variants (single, batched, strided) used to illustrate launch amortization and memory coalescing."),
        ("`compare.py`, `workload_config.py`, `arch_config.py`, `expectations_gb10.json`", "Harness entrypoint, workload shapes, architecture overrides, and stored expectation thresholds."),
    ],
    validation=[
        "`python compare.py` reports optimized_performance achieving >=2x tokens/sec vs the baseline on default microbatch sizes.",
        "Running `make && ./baseline_gemm_sm121` vs `optimized_gemm_batched_sm121` shows a substantial drop in launch count and total runtime.",
        "`python baseline_guided_decoding_math.py --validate` and the optimized twin agree numerically for every request.",
    ],
    notes=[
        "`requirements.txt` pins lightweight extras (Typer, tabulate) used by helper scripts.",
        "`Makefile` builds the CUDA GEMM binaries with SM-specific suffixes for quick diffing.",
    ],
)

ENTRIES["ch2"] = chapter_entry(
    slug="ch2",
    title="Chapter 2 - GPU Hardware Architecture",
    summary=dedent(
        """\
        Provides architecture awareness tooling for Blackwell-era systems-query SM and memory specs, validate NVLink throughput, and experiment with CPU-GPU coherency so optimizations stay grounded in measured hardware limits."""
    ),
    goals=[
        "Query and log GPU, CPU, and fabric capabilities before running performance studies.",
        "Measure NVLink, PCIe, and memory-bandwidth ceilings using purpose-built microbenchmarks.",
        "Validate Grace-Blackwell coherency paths to know when zero-copy buffers help or hurt.",
        "Contrast baseline vs optimized cuBLAS invocations to highlight architecture-specific tuning levers.",
    ],
    contents=[
        ("`hardware_info.py`, `cpu_gpu_topology_aware.py`", "System scanners that record GPU capabilities, NUMA layout, NVLink/NVSwitch connectivity, and affinity hints."),
        ("`nvlink_c2c_bandwidth_benchmark.py`, `baseline_memory_transfer.py`, `optimized_memory_transfer.py`, `optimized_memory_transfer_nvlink.cu`, `optimized_memory_transfer_zero_copy.cu`", "Peer-to-peer and zero-copy experiments for quantifying NVLink, PCIe, and coherent memory performance."),
        ("`cpu_gpu_grace_blackwell_coherency.cu`, `cpu_gpu_grace_blackwell_coherency_sm121`", "Grace-Blackwell cache-coherent samples that compare explicit transfers vs shared mappings."),
        ("`baseline_cublas.py`, `optimized_cublas.py`", "cuBLAS GEMM benchmark pair that toggles TF32, tensor op math, and stream affinity to highlight architecture knobs."),
        ("`compare.py`, `Makefile`, `expectations_gb10.json`", "Harness driver, CUDA build rules, and expectation file for automated pass/fail checks."),
    ],
    validation=[
        "`python hardware_info.py --json artifacts/hw_snapshot.json` records the correct device name, SM count, and HBM size for every GPU in the system.",
        "`python nvlink_c2c_bandwidth_benchmark.py --gpus 0 1` sustains ~250 GB/s unidirectional on NVLink-connected pairs; mismatches flag topology or driver issues.",
        "Running the coherency sample shows zero-copy benefiting sub-MB transfers while large transfers favor explicit H2D copies, matching the documented thresholds.",
    ],
    notes=[
        "Grace-only coherency tests require GB200/GB300 nodes; the binaries no-op on PCIe-only hosts.",
        "`Makefile` builds both CUDA and CPU tools so results can be compared without leaving the chapter.",
    ],
)

ENTRIES["ch3"] = chapter_entry(
    slug="ch3",
    title="Chapter 3 - System Tuning",
    summary=dedent(
        """\
        Captures the host-level changes-NUMA pinning, governor tweaks, container settings, and Kubernetes manifests-that keep GPU workloads fed before kernel-level optimization begins."""
    ),
    goals=[
        "Diagnose CPU and memory affinity issues that throttle GPU pipelines.",
        "Harden Docker and Kubernetes environments for sustained GPU throughput on shared clusters.",
        "Automate repeatable system tuning via shell scripts so lab machines stay consistent.",
        "Quantify how host-level fixes raise GEMM throughput and reduce launch latency.",
    ],
    contents=[
        ("`baseline_numa_unaware.py`, `optimized_numa_unaware.py`, `bind_numa_affinity.py`, `numa_topology_script.sh`", "NUMA diagnostics and pinning helpers for binding data loaders, NCCL ranks, and GPU contexts to the right CPU sockets."),
        ("`baseline_docker.py`, `optimized_docker.py`, `docker_gpu_optimized.dockerfile`, `system_tuning.sh`, `gpu_setup_commands.sh`", "Container configs plus host setup scripts that toggle persistence mode, huge pages, IRQ steering, and MIG visibility."),
        ("`baseline_kubernetes.py`, `optimized_kubernetes.py`, `kubernetes_mig_pod.yaml`, `kubernetes_topology_pod.yaml`", "Kubernetes manifests demonstrating topology-aware scheduling and MIG partitioning for multi-tenant fleets."),
        ("`cpu_gpu_numa_optimizations.sh`, `system_tuning.sh`, `gpu_setup_commands.sh`", "Workflow scripts for aligning CPU governors, cgroup limits, persistence mode, and driver settings with the benchmark harness."),
        ("`baseline_gemm.py`, `optimized_gemm.py`, `train.py`", "Simple GEMM + training loops that surface the impact of system tuning changes in measurable FLOP/s."),
        ("`compare.py`, `requirements.txt`, `expectations_gb10.json`", "Harness entry, Python deps, and regression thresholds."),
    ],
    validation=[
        "Run `python baseline_numa_unaware.py --diagnostics` before and after `bind_numa_affinity.py` to ensure cross-socket memory traffic drops to near zero.",
        "`python optimized_docker.py --image docker_gpu_optimized.dockerfile` should sustain the same throughput as host runs while keeping GPU clocks pinned.",
        "`python compare.py --examples gemm` shows optimized_gemm matching the measured host peak after applying `system_tuning.sh`.",
    ],
    notes=[
        "`cpu_gpu_numa_optimizations.sh` is safe to rerun after every reboot; it re-applies irqbalance pinning and governor settings.",
        "Kubernetes manifests document the necessary annotations for NVLink/NVSwitch affinity without pointing to external repos.",
    ],
)

ENTRIES["ch4"] = chapter_entry(
    slug="ch4",
    title="Chapter 4 - Multi-GPU Distribution",
    summary=dedent(
        """\
        Demonstrates how to scale training and inference across multiple Blackwell GPUs with NVLink/NVSwitch fabric awareness, NCCL tuning, NVSHMEM collectives, and symmetric memory patterns."""
    ),
    goals=[
        "Benchmark data-parallel and tensor-parallel training loops with and without overlap.",
        "Quantify NVLink bandwidth and topology effects when mixing local and disaggregated GPUs.",
        "Experiment with NVSHMEM pipelines to reduce host involvement in GPU synchronization.",
        "Adopt symmetric memory pools to simplify KV-cache replication and optimizer state sharding.",
    ],
    contents=[
        ("`baseline_dataparallel.py`, `optimized_dataparallel.py`, `baseline_dataparallel_basic.py`, `optimized_dataparallel_basic.py`", "Reference data-parallel loops that compare naive gradient exchange vs fused+overlapped NCCL."),
        ("`baseline_no_overlap.py`, `optimized_no_overlap.py`, `baseline_no_overlap_basic.py`, `optimized_no_overlap_basic.py`", "Overlap studies that stage compute/comm concurrency and pipeline microbatches to hide allreduce latency."),
        ("`baseline_nvlink.py`, `optimized_nvlink.py`, `baseline_nvlink_multigpu.py`, `optimized_nvlink_multigpu.py`, `baseline_nvlink_topology_blind.py`, `optimized_nvlink_topology_aware.py`", "NVLink and NVSwitch exercises for validating peer bandwidth and NUMA-aware rank placement."),
        ("`baseline_continuous_batching_multigpu.py`, `optimized_continuous_batching_multigpu.py`, `baseline_disaggregated.py`, `optimized_disaggregated.py`", "Continuous batching + disaggregated inference demos that showcase NVLink pooling and remote KV reuse."),
        ("`baseline_nvshmem_pipeline_parallel_multigpu.py`, `optimized_nvshmem_pipeline_parallel_multigpu.py`, `baseline_nvshmem_training_example_multigpu.py`, `optimized_nvshmem_training_example_multigpu.py`", "NVSHMEM pipeline and training samples highlighting device-driven synchronization benefits."),
        ("`baseline_symmetric_memory_multigpu.py`, `optimized_symmetric_memory_multigpu.py`, `baseline_symmetric_memory_perf.py`, `optimized_symmetric_memory_perf.py`", "Symmetric memory utilities for distributed KV cache and optimizer shards."),
        ("`compare.py`, `requirements.txt`, `expectations_gb10.json`, `bandwidth_benchmark_suite_multigpu.py`, `nccl_benchmark.py`", "Harness driver plus standalone NCCL/NVLink sweepers for topology bring-up."),
    ],
    validation=[
        "`python compare.py --examples dataparallel` shows the optimized pair overlapping compute and communication with lower latency.",
        "`python bandwidth_benchmark_suite_multigpu.py --profile minimal` surfaces >=250 GB/s links on connected GPU pairs and highlights any slow hops.",
        "NVSHMEM samples emit consistent outputs when `NVSHMEM_SYMMETRIC_SIZE` is sized to hold the workload; mismatched config raises clear errors.",
    ],
    notes=[
        "`symmetric_memory_*` helpers hold user-space allocators for pooling KV-cache lines across GPUs without NVSwitch penalties.",
        "Use `nccl_blackwell_config.py` to seed NCCL env vars (min NRings, IB mapping) before launching multi-node tests.",
    ],
)

ENTRIES["ch5"] = chapter_entry(
    slug="ch5",
    title="Chapter 5 - Storage and IO Optimization",
    summary=dedent(
        """\
        Focuses on feeding GPUs efficiently: tune DataLoader workers, vectorize preprocessing, overlap IO with compute, and adopt GPUDirect Storage when NVMe traffic becomes the bottleneck."""
    ),
    goals=[
        "Detect IO stalls via harness metrics and restructure pipelines to keep GPUs busy.",
        "Tune PyTorch DataLoader knobs (workers, prefetch, pinned memory) for large-batch training.",
        "Evaluate GPUDirect Storage paths vs traditional CPU-mediated reads.",
        "Benchmark remote storage and distributed data reading strategies.",
    ],
    contents=[
        ("`baseline_storage_cpu.py`, `optimized_storage_cpu.py`", "Single-node dataloader comparison covering worker count, pinned memory, and caching strategies."),
        ("`baseline_vectorization.py`, `optimized_vectorization.py`", "Vectorized parsing and memory-map examples that remove Python loops from preprocessing."),
        ("`baseline_ai.py`, `optimized_ai.py`, `storage_io_optimization.py`", "LLM-style token pipelines showcasing overlapping compute with streaming reads and prefetch."),
        ("`baseline_distributed.py`, `optimized_distributed.py`", "Multi-node reader pair demonstrating sharded datasets and rendezvous barriers."),
        ("`gds_cufile_minimal.py`, `gpudirect_storage_example.py`", "GPUDirect Storage samples for verifying cuFile setup, buffer alignment, and throughput."),
        ("`compare.py`, `requirements.txt`, `expectations_gb10.json`", "Harness entrypoint plus expectation baselines for spotting regressions."),
    ],
    validation=[
        "`python baseline_storage_cpu.py --inspect` exposes CPU wait time > GPU time; `optimized_storage_cpu.py` reverses the ratio with >=80% GPU utilization.",
        "`python gds_cufile_minimal.py --bytes 1073741824` sustains multi-GB/s throughput when `/etc/cufile.json` is configured and NVMe advertises GPUDirect support.",
        "`python compare.py --examples ai` shows optimized_ai eliminating CPU-side preprocessing from the critical path.",
    ],
    notes=[
        "GPUDirect scripts fall back to host-mediated reads when `libcufile.so` is unavailable, making it safe to run on dev laptops.",
        "`requirements.txt` captures the limited extra deps (like `lmdb`) needed for the dataset shims.",
    ],
)

ENTRIES["ch6"] = chapter_entry(
    slug="ch6",
    title="Chapter 6 - CUDA Programming Fundamentals",
    summary=dedent(
        """\
        Moves from Python into CUDA C++: write first kernels, reason about occupancy, control memory layouts, and experiment with ILP, launch bounds, and unified memory on Blackwell devices."""
    ),
    goals=[
        "Write and launch custom kernels that mirror the harness workloads.",
        "Understand how occupancy, launch bounds, and register pressure interact.",
        "Use ILP and vectorized memory ops to unlock more throughput per thread.",
        "Validate unified memory and allocator tuning on Blackwell GPUs.",
    ],
    contents=[
        ("`my_first_kernel.cu`, `simple_kernel.cu`, `baseline_add.cu`, `optimized_add.cu`, `baseline_add.py`, `optimized_add.py`", "Hello-world kernels plus Python wrappers for verifying CUDA build chains and launch parameters."),
        ("`baseline_add_tensors.cu`, `optimized_add_tensors.cu`, `baseline_add_tensors.py`, `optimized_add_tensors.py`", "Tensor-oriented adds with automatic pinned-memory staging and correctness checks."),
        ("`baseline_attention_ilp.py`, `baseline_gemm_ilp.py`, `optimized_gemm_ilp.py`, `optimized_ilp_low_occupancy_vec4.cu`, `optimized_ilp_extreme_low_occupancy_vec4.cu`", "Instruction-level parallelism studies that manipulate loop unrolling, registers, and vector width."),
        ("`baseline_bank_conflicts.cu`, `optimized_bank_conflicts.cu`, `baseline_launch_bounds*.{py,cu}`, `optimized_launch_bounds*.{py,cu}`", "Bank conflict and launch-bound exercises to highlight shared memory layouts and CTA sizing."),
        ("`baseline_autotuning.py`, `optimized_autotuning.py`, `memory_pool_tuning.cu`, `stream_ordered_allocator/`", "Autotuning harness plus allocator experiments for controlling fragmentation and stream ordering."),
        ("`unified_memory.cu`, `occupancy_api.cu`, `baseline_quantization_ilp.py`, `optimized_quantization_ilp.py`", "Unified memory demo, occupancy calculator sample, and quantization-focused ILP workloads."),
        ("`compare.py`, `Makefile`, `expectations_gb10.json`, `workload_config.py`", "Harness entry, build scripts, expectation baselines, and workload settings."),
    ],
    validation=[
        "`nvcc -o baseline_add_sm121 baseline_add.cu` vs the optimized vectorized version shows a clear bandwidth delta when inspected with Nsight Compute.",
        "`python optimized_autotuning.py --search` converges to the same schedule as the curated preset and logs the score table under `artifacts/`.",
        "`python compare.py --examples ilp` confirms optimized ILP kernels achieving higher instructions-per-byte with identical outputs.",
    ],
    notes=[
        "`arch_config.py` forces SM-specific compile flags (e.g., disabling pipelines on unsupported GPUs) so targets fail gracefully on older hardware.",
        "CUDA extensions in `cuda_extensions/` can be imported directly into notebooks for interactive prototyping.",
    ],
)

ENTRIES["ch7"] = chapter_entry(
    slug="ch7",
    title="Chapter 7 - Memory Access Patterns",
    summary=dedent(
        """\
        Teaches how memory layout drives performance: coalesced copies, tiled matmuls, async prefetch, TMA transfers, and shared-memory staging for lookup-heavy workloads."""
    ),
    goals=[
        "Measure the gap between scalar, coalesced, and vectorized memory moves.",
        "Use shared-memory tiling, TMA, and async copy to keep tensor cores saturated.",
        "Analyze lookup-heavy workloads and mitigate cache-thrashing access patterns.",
        "Quantify transpose and gather/scatter penalties to justify layout changes.",
    ],
    contents=[
        ("`baseline_copy_scalar.cu`, `baseline_copy_uncoalesced.cu`, `baseline_uncoalesced_copy.py`, `optimized_copy_coalesced.cu`, `optimized_copy_vectorized.cu`, `optimized_copy_vectorized_sm121`", "Copy kernels highlighting coalescing, vector width, and warp-level efficiency."),
        ("`baseline_hbm3e_copy.cu`, `baseline_hbm3e_peak.cu`, `optimized_hbm3e_copy.cu`, `optimized_hbm3e_peak.cu`, `baseline_hbm3ecopy.py`, `optimized_hbm3ecopy.py`", "HBM3e peak-bandwidth probes with CUDA and Python harnesses."),
        ("`baseline_async_prefetch.cu`, `optimized_async_prefetch.cu`, `baseline_tma_copy.cu`, `baseline_tma_copy.py`, `optimized_async_prefetch.py`", "Async/TMA samples that overlap global-memory fetch with computation."),
        ("`baseline_matmul.cu`, `baseline_matmul_cuda.py`, `optimized_matmul_cuda.py`, `optimized_matmul_tiled.cu`", "Matmul implementations to contrast naive global-memory access with shared-memory tiling and warp-level reuse."),
        ("`baseline_lookup.cu`, `baseline_lookup.py`, `optimized_lookup.cu`, `lookup_pytorch.py`", "Cache-sensitive lookup workloads demonstrating how to reorganize tables for better locality."),
        ("`baseline_transpose.cu`, `baseline_transpose.py`, `optimized_copy_vectorized.cu`, `optimized_transpose.py`", "Transpose and gather/scatter experiments that show how to minimize bank conflicts."),
        ("`compare.py`, `Makefile`, `expectations_gb10.json`, `memory_access_pytorch.py`", "Harness entry, build recipes, expectation thresholds, and PyTorch validation scripts."),
    ],
    validation=[
        "`python baseline_hbm3ecopy.py --bytes 1073741824` reports noticeably lower GB/s than `optimized_hbm3ecopy.py`, proving vectorization plus async copies work.",
        "`python compare.py --examples async_prefetch` shows optimized_async_prefetch reducing total kernel count while preserving accuracy.",
        "Nsight Compute captures of `optimized_matmul_tiled.cu` hit >80% shared-memory bandwidth utilization with minimal bank conflicts.",
    ],
    notes=[
        "Toggle `TORCH_COMPILE_MODE` when using the Python matmul wrappers to verify fusion benefits alongside the raw CUDA kernels.",
        "HBM tooling reads real peak numbers from `benchmark_peak_results_*.json` when present, providing realistic reference ceilings.",
    ],
)

ENTRIES["ch8"] = chapter_entry(
    slug="ch8",
    title="Chapter 8 - Occupancy & Pipeline Tuning",
    summary=dedent(
        """\
        Concentrates on resource balancing: adjust block sizes, registers, and shared memory to keep SMs full while hiding TMEM latency via double buffering, loop unrolling, and async pipelines."""
    ),
    goals=[
        "Tune occupancy explicitly and observe how register counts limit resident CTAs.",
        "Apply double buffering and async staging to overlap DRAM fetch with compute.",
        "Use tiling, loop unrolling, and AI-specific thresholds to control latency vs throughput.",
        "Measure how pipelined schedules change SM/TMEM utilization using the shared harness.",
    ],
    contents=[
        ("`baseline_occupancy_tuning.py`, `optimized_occupancy_tuning.py`, `optimized_occupancy_tuning_bs64.py`, `optimized_occupancy_tuning_maxrreg32.py`, `occupancy_api_example.cu`, `occupancy_tuning.cu`", "Occupancy studies that sweep CTA shapes, register caps, and API-computed limits."),
        ("`baseline_double_buffering.cu`, `baseline_double_buffering.py`, `optimized_double_buffering_pipelined.cu`, `optimized_double_buffering.py`, `double_buffering_kernels.cu`", "Double-buffered kernels and their Python drivers showing how to keep tensor cores busy."),
        ("`baseline_hbm.cu`, `baseline_hbm.py`, `optimized_hbm.py`, `optimized_hbm_vectorized.cu`, `hbm_kernels.cu`", "HBM streaming workloads that compare scalar, vectorized, and asynchronous fetch patterns."),
        ("`baseline_loop_unrolling.cu`, `baseline_loop_unrolling.py`, `optimized_loop_unrolling.cu`, `optimized_loop_unrolling.py`, `loop_unrolling_kernels.cu`", "Loop unrolling case studies targeting various ILP regimes."),
        ("`baseline_threshold.py`, `baseline_thresholdtma.py`, `optimized_threshold.py`, `optimized_thresholdtma.py`, `threshold_kernels.cu`, `threshold_tma_benchmark_base.py`", "Threshold operators implemented with scalar, vectorized, and TMA-backed pipelines."),
        ("`baseline_tiling.py`, `baseline_tiling_tcgen05.py`, `optimized_tiling.py`, `optimized_tiling_tcgen05.py`, `tiling_kernels.cu`, `tiling_extension_tcgen05.py`", "Tile schedulers for tcgen05 matmuls, including safe fallbacks when tcgen05 isn't available."),
        ("`compare.py`, `requirements.txt`, `expectations_gb10.json`, `ai_optimization_kernels.cu`", "Harness entry, dependencies, regression thresholds, and AI-optimization helper kernels."),
    ],
    validation=[
        "Nsight Compute traces for `optimized_double_buffering_pipelined.cu` should show overlapping smem/ldgst transactions with minimal idle cycles.",
        "`python optimized_occupancy_tuning.py --report` emits CTA/block configurations hitting the highest occupancy without exceeding register limits.",
        "`python compare.py --examples threshold` confirms the TMA-backed kernels reducing latency vs scalar reference implementations.",
    ],
    notes=[
        "`arch_config.py` exposes toggles for enabling/disabling tcgen05 lowering per GPU so the same scripts work on SM100 and SM121.",
        "`build/` caches CUDA object files per configuration; clean via `python cleanup.py --include-build` when adjusting toolchains.",
    ],
)

ENTRIES["ch9"] = chapter_entry(
    slug="ch9",
    title="Chapter 9 - Arithmetic Intensity & Kernel Fusion",
    summary=dedent(
        """\
        Explores how to move workloads along the roofline: raise arithmetic intensity with tiling, fuse memory-bound kernels, and deploy CUTLASS/Triton/inline-PTX paths built for Blackwell tensor cores."""
    ),
    goals=[
        "Separate compute-bound vs memory-bound behaviors and adjust kernels accordingly.",
        "Design micro-tiling schedules that balance register pressure with data reuse.",
        "Leverage CUTLASS and Triton for rapid iteration while keeping custom CUDA fallbacks.",
        "Fuse reduction-heavy kernels (e.g., norm + activation) to eliminate redundant memory trips.",
    ],
    contents=[
        ("`baseline_compute_bound.py`, `optimized_compute_bound.py`, `baseline_memory_bound.py`, `optimized_memory_bound.py`", "Reference kernels that isolate compute vs bandwidth ceilings and demonstrate tuning strategies."),
        ("`baseline_micro_tiling_matmul.cu`, `baseline_micro_tiling_matmul.py`, `optimized_micro_tiling_matmul.cu`, `optimized_micro_tiling_matmul.py`", "Micro-tiling matmuls with explicit register blocking and cp.async prefetch."),
        ("`baseline_cutlass_gemm.cu`, `baseline_cutlass_gemm.py`, `optimized_cutlass_gemm.cu`, `optimized_cutlass_gemm.py`, `warp_specialized_cuda.cu`", "CUTLASS-driven matmuls and warp-specialized kernels showcasing tcgen05 lowering and occupancy tuning."),
        ("`baseline_fused_l2norm.cu`, `baseline_fused_l2norm.py`, `optimized_fused_l2norm.cu`, `optimized_fused_l2norm.py`, `fusedL2Norm/`", "Fusion examples that merge L2 norm + scaling while staying numerically stable."),
        ("`baseline_triton.py`, `optimized_triton.py`, `warp_specialized_triton.py`", "Triton counterparts for quick prototyping and verifying compiler-generated PTX on Blackwell."),
        ("`baseline_warp_specialization_producer_consumer.py`, `optimized_warp_specialization_producer_consumer.py`, `two_stage_pipeline.cu`", "Warp cooperation samples emphasizing producer/consumer pipelines and inline PTX hooks."),
        ("`compare.py`, `requirements.txt`, `expectations_gb10.json`", "Harness hooks plus regression thresholds for every example."),
    ],
    validation=[
        "`python baseline_compute_bound.py --summaries` reports much higher arithmetic intensity than `baseline_memory_bound.py`, matching the roofline plots.",
        "`python optimized_cutlass_gemm.py --sizes 4096 4096 8192` hits TFLOP/s numbers close to `benchmark_peak_results.json` for the same device.",
        "`python compare.py --examples fused_l2norm` confirms numerically identical outputs before and after fusion.",
    ],
    notes=[
        "`inline_ptx_example.cu` demonstrates how to wrap tcgen05 intrinsics safely with architecture guards.",
        "`requirements.txt` includes Triton nightly pinning so the kernels track PyTorch 2.10-dev features.",
    ],
)

ENTRIES["ch10"] = chapter_entry(
    slug="ch10",
    title="Chapter 10 - Tensor Core Pipelines & Cluster Features",
    summary=dedent(
        """\
        Applies tensor-core friendly scheduling on Blackwell: warp specialization, TMA-powered pipelines, persistent kernels, and thread-block clusters with DSMEM and NVLink-C2C awareness."""
    ),
    goals=[
        "Use warp specialization and cp.async/TMA to keep tensor cores saturated.",
        "Prototype persistent matmuls that amortize launch overhead across iterations.",
        "Exercise thread-block clusters with and without DSMEM to understand hardware limits.",
        "Combine PyTorch, Triton, and CUDA kernels while keeping expectations synchronized.",
    ],
    contents=[
        ("`baseline_attention.py`, `optimized_attention.py`, `baseline_flash_attention.py`, `optimized_flash_attention.py`, `analyze_scaling.py`", "Attention workloads that span eager, fused, and `torch.compile` paths for modern decoder models."),
        ("`baseline_batch.py`, `optimized_batch.py`, `baseline_matmul.py`, `optimized_matmul.py`, `baseline_matmul_tcgen05.py`, `optimized_matmul_tcgen05.py`", "Tensor-core matmul variants demonstrating tcgen05 lowering, register tiling, and PyTorch integration."),
        ("`baseline_double_buffered_pipeline.{py,cu}`, `optimized_double_buffered_pipeline.{py,cu}`, `baseline_tma_2d_pipeline.py`, `optimized_tma_2d_pipeline.py`", "Async pipeline samples mixing cp.async, TMA, and manual double buffering."),
        ("`baseline_cluster_group*.{py,cu}`, `optimized_cluster_group*.{py,cu}`, `cluster_group_common.cuh`, `cluster_group_utils.py`", "Clustered kernel suite covering DSMEM-enabled and DSMEM-free thread-block clusters."),
        ("`baseline_cooperative_persistent.{py,cu}`, `optimized_cooperative_persistent.{py,cu}`, `baseline_persistent_matmul_tma.py`, `optimized_persistent_matmul_tma.py`", "Persistent kernels combining cooperative groups with TMA streams for steady-state throughput."),
        ("`baseline_flash_attn_tma_micro_pipeline.{py,cu}`, `optimized_flash_attn_tma_micro_pipeline.{py,cu}`, `baseline_warp_specialized_pipeline*.{py,cu}`, `optimized_warp_specialized_pipeline*.{py,cu}`", "Micro-pipeline and warp specialization studies that mix Triton, CUDA, and inline PTX."),
        ("`compare.py`, `workload_config.py`, `perf_sweep.sh`, `profile.sh`, `requirements_cufile.txt`", "Harness entry, workload dials, performance sweeps, Nsight automation, and optional cuFile deps."),
    ],
    validation=[
        "Cluster-enabled kernels fail fast on hardware without DSMEM support, while DSMEM-free variants still execute-use this to confirm cluster capability flags.",
        "`python optimized_flash_attn_tma_micro_pipeline.py --profile` produces fewer kernel launches and higher achieved FLOP/s than the baseline script.",
        "`python perf_sweep.sh --chapter ch10` records TFLOP/s vs batch size curves, proving persistent kernels amortize launch costs.",
    ],
    notes=[
        "`cufile_gds_example.py` demonstrates integrating GPUDirect Storage into tensor-core pipelines for IO-heavy training loops.",
        "`requirements_cufile.txt` holds the optional `cufile` wheel; install it only on hosts with GPUDirect Storage enabled.",
    ],
)

ENTRIES["ch11"] = chapter_entry(
    slug="ch11",
    title="Chapter 11 - Streams & Concurrency",
    summary=dedent(
        """\
        Explains how to overlap compute, memory, and communication on Blackwell using CUDA streams, ordered sequences, Hyper-Q, warp-specialized pipelines, and adaptive scheduling."""
    ),
    goals=[
        "Use multiple CUDA streams to overlap independent kernels without starving priority work.",
        "Control ordering constraints for KV-cache updates and stream-ordered memory pools.",
        "Benchmark warp-specialized multistream kernels that share data via DSMEM.",
        "Introduce adaptive policies that adjust stream usage based on runtime telemetry.",
    ],
    contents=[
        ("`baseline_streams.py`, `optimized_streams.py`, `baseline_streams.cu`, `optimized_streams_ordered.cu`, `stream_overlap_base.py`", "Core stream overlap demos that contrast serialized launches with overlapped workloads."),
        ("`baseline_stream_ordered.py`, `baseline_stream_ordered_kv_cache.py`, `optimized_stream_ordered.py`, `optimized_stream_ordered_kv_cache.py`", "Stream-ordered allocator and KV-cache examples ensuring deterministic updates while enabling overlap."),
        ("`baseline_gemm_streams.py`, `optimized_gemm_streams.py`, `baseline_tensor_cores_streams.py`, `optimized_tensor_cores_streams.py`", "GEMM pipelines that schedule tensor-core kernels across multiple streams to decouple math vs IO phases."),
        ("`baseline_distributed_streams.py`, `optimized_distributed_streams.py`, `baseline_adaptive_streams.py`, `optimized_adaptive_streams.py`", "Adaptive streaming controllers that balance NCCL, compute, and IO tasks on large systems."),
        ("`baseline_warp_specialization_multistream.*`, `optimized_warp_specialized_multistream.*`, `warp_specialized_cluster_pipeline_multistream.cu`", "Warp-specialized multistream kernels demonstrating DSMEM usage and per-stream specialization."),
        ("`compare.py`, `Makefile`, `expectations_gb10.json`", "Harness driver plus expectation data for concurrency regressions."),
    ],
    validation=[
        "`python optimized_streams.py --trace` captures overlapping NVTX ranges in Nsight Systems, proving concurrency is active.",
        "`python optimized_stream_ordered_kv_cache.py --validate` matches the baseline's outputs while reducing idle gaps between cache updates.",
        "Warp-specialized multistream kernels flag unsupported hardware (missing DSMEM) immediately, preventing silent fallbacks.",
    ],
    notes=[
        "`warp_specialized_triton.py` provides a Triton analogue for the CUDA concurrency demos so you can compare compiler-generated schedules.",
        "`optimized_kv_prefetch_pipeline_enhanced.cu` builds on the DSMEM kernels bundled in this directory so you can study the entire pipeline locally.",
    ],
)

ENTRIES["ch12"] = chapter_entry(
    slug="ch12",
    title="Chapter 12 - CUDA Graphs & Dynamic Workloads",
    summary=dedent(
        """\
        Covers modern CUDA Graph capabilities-conditional capture, graph memory tuning, dynamic parallelism, and work queues-to keep irregular workloads performant without per-launch overhead."""
    ),
    goals=[
        "Capture steady-state workloads into CUDA Graphs and study the delta vs eager launches.",
        "Use conditional nodes and graph memory pools for adaptive pipelines.",
        "Experiment with device-side launches (dynamic parallelism) to reduce CPU involvement.",
        "Implement GPU-resident work queues and uneven partition schedulers.",
    ],
    contents=[
        ("`baseline_cuda_graphs.py`, `optimized_cuda_graphs.py`, `baseline_cuda_graphs_conditional*.cu`, `optimized_cuda_graphs_conditional*.cu`", "Graph capture demos that evolve from simple replay to conditional and DSM-aware execution."),
        ("`baseline_graph_bandwidth.{py,cu}`, `optimized_graph_bandwidth.{py,cu}`, `baseline_kernel_launches.py`, `optimized_kernel_launches.py`", "Launch- and bandwidth-focused studies illustrating how graphs reduce CPU overhead."),
        ("`baseline_dynamic_parallelism_host.cu`, `baseline_dynamic_parallelism_device.cu`, `optimized_dynamic_parallelism_host.cu`, `optimized_dynamic_parallelism_device.cu`, `dynamic_parallelism_sm121/`", "Device-side launch samples showing when dynamic parallelism helps or hurts."),
        ("`baseline_work_queue.{py,cu}`, `optimized_work_queue.{py,cu}`, `work_queue_common.cuh`", "GPU work queues for irregular batch sizes, including NVTX instrumentation."),
        ("`baseline_uneven_partition.cu`, `optimized_uneven_partition.cu`, `baseline_uneven_static.cu`, `optimized_uneven_static.cu`", "Uneven workload partitioners that rebalance CTA assignments at runtime."),
        ("`baseline_kernel_fusion.{py,cu}`, `optimized_kernel_fusion.{py,cu}`", "Kernel fusion exercises within graph capture so you can remove CPU synchronization entirely."),
        ("`compare.py`, `cuda_extensions/`, `expectations_gb10.json`", "Harness entry, extension stubs, and expectation thresholds."),
    ],
    validation=[
        "`python optimized_cuda_graphs.py --iterations 100` should report lower wall-clock time than the baseline while matching outputs.",
        "Device-side dynamic parallelism samples emit warnings on unsupported hardware, ensuring you only trust data from GPUs with the feature enabled.",
        "`python optimized_work_queue.py --trace` exposes balanced dequeue times across CTAs when compared to the baseline's stragglers.",
    ],
    notes=[
        "`cuda_graphs_workload.cuh` holds reusable graph capture helpers when you want to wrap your own kernels.",
        "`helper_*.cu` files contain host/device glue for the dynamic-parallelism case studies-copy them when bootstrapping new experiments.",
    ],
)

ENTRIES["ch13"] = chapter_entry(
    slug="ch13",
    title="Chapter 13 - PyTorch Profiling & Memory Tuning",
    summary=dedent(
        """\
        Focuses on PyTorch-centric optimizations: compiled autograd, memory profiling, FSDP/context/expert parallelism, and FP8/quantization workflows backed by the same harness infrastructure."""
    ),
    goals=[
        "Profile PyTorch training loops end-to-end, capturing goodput, memory, and kernel traces.",
        "Apply `torch.compile`, regional compilation, and custom allocators to reduce overhead.",
        "Tune DataLoader, KV-cache, and optimizer states to eliminate fragmentation.",
        "Exercise FP8/quantized training recipes with Transformer Engine integration.",
    ],
    contents=[
        ("`baseline_training_standard.py`, `optimized_training_standard.py`, `train.py`, `train_deepseek_v3.py`, `train_deepseek_coder.py`", "Reference training loops showcasing eager vs compiled paths and DeepSeek-inspired configs."),
        ("`baseline_dataloader_default.py`, `optimized_dataloader_default.py`, `baseline_memory_profiling.py`, `optimized_memory_profiling.py`, `memory_profiling.py`", "DataLoader/memory studies that explain how to read allocator stats and fix leaks."),
        ("`baseline_attention_standard.py`, `optimized_attention_standard.py`, `baseline_arithmetic_intensity.py`, `optimized_arithmetic_intensity.py`, `baseline_matmul_pytorch.py`, `optimized_matmul_pytorch.py`", "Attention and matmul microbenchmarks tuned purely within PyTorch."),
        ("`baseline_context_parallelism.py`, `optimized_context_parallelism.py`, `baseline_expert_parallelism.py`, `optimized_expert_parallelism.py`, `fsdp_example.py`", "Context, expert, and FSDP sharding demos for scaling beyond a single GPU."),
        ("`baseline_precisionfp8*.py`, `optimized_precisionfp8*.py`, `baseline_precisionmixed.py`, `optimized_precisionmixed.py`, `compiled_autograd.py`", "Precision-management suites covering Transformer Engine and compiled autograd recipes."),
        ("`baseline_quantization.py`, `optimized_quantization.py`, `baseline_kv_cache_naive.py`, `optimized_kv_cache.py`", "Quantization and KV-cache pipelines for inference/training memory savings."),
        ("`compare.py`, `compare_perf.py`, `requirements.txt`, `expectations_gb10.json`, `workload_config.py`", "Harness entry, performance comparison helper, dependencies, and regression baselines."),
    ],
    validation=[
        "`python compare.py --examples training_standard` shows optimized training runs producing higher goodput with identical metrics.",
        "`python optimized_precisionfp8_te.py --validate` confirms Transformer Engine calibration plus NVFP8 execution with max error tolerances enforced.",
        "`python memory_profiling.py --dump` and the optimized variant demonstrate allocator fragmentation dropping after applying the recommended knobs.",
    ],
    notes=[
        "`custom_allocator.py` contains a standalone torch allocator shim that can be re-used in other chapters when debugging fragmentation.",
        "`compiled_autograd.py` doubles as a tutorial on partial graph capture; the README here references it directly.",
    ],
)

ENTRIES["ch14"] = chapter_entry(
    slug="ch14",
    title="Chapter 14 - Compiler & Triton Optimization",
    summary=dedent(
        """\
        Highlights compiler-driven acceleration: `torch.compile` workflows, Triton kernels, CUTLASS/TMA experimentation, and quantization-aware communication, all validated through the shared harness."""
    ),
    goals=[
        "Adopt `torch.compile` modes for large models while tracking compile-time and steady-state gains.",
        "Author Triton kernels (including TMA schedules) that rival custom CUDA.",
        "Profile FlexAttention and regional compilation strategies end-to-end.",
        "Blend quantization with NCCL and pipeline overlap without regressions.",
    ],
    contents=[
        ("`baseline_model_eager.py`, `optimized_model_eager.py`, `torch_compile_large_model.py`, `torch_compiler_examples.py`, `training_large_model_1_5x.py`", "Model-scale examples showcasing compile modes, guard rails, and large-model sanity tests."),
        ("`baseline_cutlass.py`, `optimized_cutlass.py`, `triton_examples.py`, `triton_tma_blackwell.py`, `triton_fp8_advanced.py`, `triton_nvshmem_example.py`", "CUTLASS vs Triton comparisons plus advanced TMA/NVSHMEM Triton kernels."),
        ("`baseline_flex_attention.py`, `optimized_flex_attention.py`, `test_flex_attention.py`", "FlexAttention workloads that validate custom score mods, masks, and compile speedups."),
        ("`baseline_nccl_quantization.py`, `optimized_nccl_quantization.py`, `deepseek_innovation_l2_bypass.py`", "Quantization-aware communication and the DeepSeek-inspired L2 bypass experiment."),
        ("`baseline_regional_triton.py`, `optimized_regional_triton.py`, `inspect_compiled_code.py`, `benchmark_tma_configs.py`", "Regional compilation and TMA parameter sweeps for auto-tuning generated kernels."),
        ("`compare.py`, `requirements.txt`, `expectations_gb10.json`, `train.py`, `transformer.py`", "Harness entry plus model definitions and dependency pins."),
    ],
    validation=[
        "`python optimized_model_eager.py --profile minimal` produces compile-time summaries followed by steady-state throughput gains vs the baseline.",
        "`python triton_tma_blackwell.py --validate` compares Triton and CUDA outputs to double-check TMA scheduling logic.",
        "`python compare.py --examples flex_attention` shows the compiled path significantly reducing kernel launch count without changing accuracy.",
    ],
    notes=[
        "`inspect_compiled_code.py` dumps Triton/PTX/Graph captures for any target; edit the helper to introspect new workloads.",
        "`requirements.txt` includes nightly Triton + PyTorch wheels to keep compiler features aligned with the CUDA 13 toolchain.",
    ],
)

ENTRIES["ch15"] = chapter_entry(
    slug="ch15",
    title="Chapter 15 - Disaggregated Inference & KV Management",
    summary=dedent(
        """\
        Addresses large-scale inference concerns: disaggregated compute/storage, KV-cache pooling over NVLink, continuous batching, and mixture-of-experts serving patterns."""
    ),
    goals=[
        "Benchmark monolithic vs disaggregated inference paths and quantify fabric costs.",
        "Design KV-cache managers that gracefully span local and remote HBM pools.",
        "Implement continuous batching and queueing so decode throughput stays high.",
        "Serve MoE models efficiently by pairing routing with optimized communication.",
    ],
    contents=[
        ("`baseline_inference_monolithic.py`, `optimized_inference_monolithic.py`, `disaggregated_inference.py`", "Single-box inference loops that establish the baseline before disaggregation."),
        ("`baseline_disaggregated_inference.py`, `optimized_disaggregated_inference.py`, `baseline_prefill_decode_disagg.py`, `optimized_prefill_decode_disagg.py`", "Disaggregated pipelines modeling remote prefills, decode overlap, and NVLink pooling."),
        ("`baseline_kv_cache_management.py`, `optimized_kv_cache_management.py`, `optimized_kv_cache_management_math.py`, `optimized_kv_cache_nvlink_pool.py`, `baseline_kv_cache_local_only.py`", "KV-cache orchestration utilities with local-only, math-only, and NVLink-pooled variants."),
        ("`baseline_continuous_batching.py`, `optimized_continuous_batching.py`", "Continuous batching scheduler demonstrating TTFT-aware queueing."),
        ("`baseline_moe_inference.py`, `optimized_moe_inference.py`", "Inference-specific MoE workloads that pair router load with communication control."),
        ("`compare.py`, `requirements.txt`, `expectations_gb10.json`, `Makefile`", "Harness entry and dependencies for inference-focused validation."),
    ],
    validation=[
        "`python optimized_disaggregated_inference.py --profile minimal` shows reduced fabric stalls compared to the baseline while maintaining accuracy parity.",
        "`python optimized_kv_cache_management.py --validate` confirms eviction + promotion policies keep decode latency within the budget.",
        "`python compare.py --examples continuous_batching` proves optimized scheduling increases tokens/sec vs naive queue draining.",
    ],
    notes=[
        "`disaggregated_inference.py` can run purely in simulation mode; set `--simulate-network` when hardware isn't wired for NVLink pooling.",
        "`Makefile` wraps the MPI/UCX targets needed for the multi-node decode experiments.",
    ],
)

ENTRIES["ch16"] = chapter_entry(
    slug="ch16",
    title="Chapter 16 - Production Inference Optimization",
    summary=dedent(
        """\
        Focuses on real-world inference services: paged attention, Flash SDP, FP8 serving, telemetry hooks, schedulers, and Blackwell-friendly load-test harnesses."""
    ),
    goals=[
        "Profile large decoder workloads to spot hotspots before deploying models.",
        "Adopt paged attention, Flash SDP, and piecewise compilation to hit latency targets.",
        "Integrate FP8 quantization, symmetric memory, and cache monitoring in serving loops.",
        "Simulate production loads (multi-node, MoE) while validating accuracy via perplexity checks.",
    ],
    contents=[
        ("`inference_optimizations_blackwell.py`, `inference_profiling.py`, `inference_server_load_test.py`, `inference_serving_8xb200.py`", "Top-level orchestration scripts for profiling and load testing multi-GPU inference deployments."),
        ("`baseline_flash_sdp.py`, `optimized_flash_sdp.py`, `baseline_paged_attention.py`, `optimized_paged_attention.py`", "Attention kernels that compare naive implementations vs Flash/paged variants."),
        ("`baseline_piece_graphs.py`, `optimized_piece_graphs.py`, `baseline_regional_compilation.py`, `optimized_regional_compilation.py`", "Piecewise graph capture and regional compilation for stable low-latency decode."),
        ("`fp8_transformer_engine.py`, `test_fp8_quantization_real.py`, `symmetric_memory_inference.py`, `multi_gpu_validation.py`", "Serving-time FP8 and symmetric-memory validations to guarantee accuracy and NVLink efficiency."),
        ("`moe_performance_benchmark.py`, `synthetic_moe_inference_benchmark.py`, `moe_workload.py`", "MoE inference harnesses that stress router placement and per-expert batching."),
        ("`cache_monitoring.py`, `dcgm_prometheus_exporter.py`, `scheduler.py`, `perplexity_eval.py`", "Telemetry, scheduling, and accuracy utilities wired into the inference pipeline."),
        ("`compare.py`, `requirements.txt`, `Makefile`, `expectations_gb10.json`", "Harness entry and dependencies for inference-focused verification."),
    ],
    validation=[
        "`python optimized_paged_attention.py --profile minimal` yields fewer page faults and improved throughput relative to the baseline script.",
        "`python symmetric_memory_inference.py --validate` confirms NVLink-backed KV replicas stay in sync with negligible skew.",
        "`python inference_server_load_test.py --duration 120` exercises the scheduler and should report stable TTFT/TPOT metrics after warm-up.",
    ],
    notes=[
        "`dcgm_prometheus_exporter.py` emits per-GPU metrics consumable by Prometheus/Grafana without extra setup.",
        "`cache_monitoring.py` can be run standalone to sanity-check allocator health between runs.",
    ],
)

ENTRIES["ch17"] = chapter_entry(
    slug="ch17",
    title="Chapter 17 - Dynamic Routing & Hybrid Serving",
    summary=dedent(
        """\
        Blends router design, disaggregated inference, and profiling discipline so Blackwell clusters can route queries between prefill/decode pools, MoE experts, and pipeline stages without sacrificing utilization."""
    ),
    goals=[
        "Implement dynamic routers that react to TTFT, TPOT, and KV-locality metrics.",
        "Profile complete inference stacks (prefill + decode) under realistic synthetic loads.",
        "Blend pipeline parallelism with routing logic for long-context workloads.",
        "Document profiling steps (roofline, Nsight) specific to the routing lab.",
    ],
    contents=[
        ("`baseline_dynamic_routing.py`, `optimized_dynamic_routing.py`, `dynamic_routing.py`, `early_rejection.py`", "Routing controllers that evolve from static heuristics to telemetry-driven admission and rejection policies."),
        ("`baseline_inference_full.py`, `optimized_inference_full.py`, `baseline_prefill_decode_disagg.py`, `optimized_prefill_decode_disagg.py`, `baseline_prefill_decode_disagg_multigpu.py`, `optimized_prefill_decode_disagg_multigpu.py`", "End-to-end inference flows modeling separate prefill and decode pools, both single-node and multi-GPU."),
        ("`baseline_pipeline_parallelism.py`, `optimized_pipeline_parallelism.py`", "Pipeline parallel workloads combining compute and KV-transfer scheduling."),
        ("`baseline_moe_router_uniform.py`, `optimized_moe_router_topology.py`, `baseline_routing_static.py`, `optimized_routing_static.py`", "Router variants for MoE and static/dynamic sharding decisions."),
        ("`baseline_memory.py`, `optimized_memory.py`, `blackwell_profiling_guide.py`, `blackwell_roofline_analysis.py`, `comprehensive_profiling_toolkit.py`", "Memory-bound case studies plus profiling guides tailored to routing workloads."),
        ("`compare.py`, `Makefile`, `expectations_gb10.json`, `dynamo_config.yaml`", "Harness entry, build rules, expectation baselines, and Dynamo config knobs."),
    ],
    validation=[
        "`python optimized_dynamic_routing.py --trace` logs TTFT/TPOT trends that settle faster than the baseline's oscillations.",
        "`python optimized_pipeline_parallelism.py --profile minimal` shows overlapping prefill/decode segments with fewer idle bubbles.",
        "`python blackwell_roofline_analysis.py --artifacts ./artifacts` reproduces the documented roofline points using your latest captures.",
    ],
    notes=[
        "`comprehensive_profiling_toolkit.py` bundles Nsight Systems/Compute runs plus summary markdown to streamline PoB updates.",
        "`baseline_prefill_decode_disagg_multigpu.py` can run in simulation-only mode by passing `--simulate-fabric`, avoiding the need for multi-node hardware while iterating on routing logic.",
    ],
)

ENTRIES["ch18"] = chapter_entry(
    slug="ch18",
    title="Chapter 18 - Advanced Attention & Decoding",
    summary=dedent(
        """\
        Collects modern decoder techniques-FlexAttention, FlexDecoding, speculative and paged attention workflows-implemented in both PyTorch and CUDA/Triton so you can iterate quickly while validating kernels on real hardware."""
    ),
    goals=[
        "Prototype FlexAttention/FlexDecoding workloads with custom masks, score mods, and KV-cache integration.",
        "Evaluate speculative decoding pipelines that trade extra compute for lower latency.",
        "Test tensor-core optimized attention kernels tailored for Blackwell tmem limits.",
        "Validate integration points with serving frameworks (vLLM) using the provided runners.",
    ],
    contents=[
        ("`baseline_flexdecoding.py`, `optimized_flexdecoding.py`, `optimized_flexdecoding_math.py`, `baseline_speculative_decoding.py`, `optimized_speculative_decoding.py`, `optimized_speculative_decoding_math.py`", "Decoding workloads comparing eager vs math-only vs optimized pipeline variants."),
        ("`baseline_tensor_cores.py`, `optimized_tensor_cores.py`, `flashmla_kernel.cu`, `warp_specialized_triton.py`", "Tensor-core attention kernels plus Triton equivalents for rapid validation."),
        ("`flex_attention_native.py`, `flex_attention_enhanced.py`, `flex_attention_large_model.py`, `kv_cache_integration_example.py`", "FlexAttention examples ranging from toy sizes to large models with KV-cache reuse."),
        ("`run_vllm_decoder.py`, `configs/`, `paged_attn_common.py`, `workload_config.py`", "Serving integrations and config presets for pushing workloads through vLLM or custom harnesses."),
        ("`compare.py`, `expectations_gb10.json`, `test_flex_attention.py`", "Harness entry, regression thresholds, and pytest coverage for FlexAttention APIs."),
    ],
    validation=[
        "`python optimized_flexdecoding.py --profiling` reports significantly fewer kernels and lower latency than the baseline while matching decoded tokens.",
        "`python run_vllm_decoder.py --config configs/flex_attention.yaml` completes with accuracy parity vs the native FlexAttention path.",
        "`python test_flex_attention.py` passes locally, confirming mask/score-mod helpers are wired correctly.",
    ],
    notes=[
        "`flex_attention` scripts accept env vars like `BLOCK_SIZE`, `DOC_SPAN`, and `SEQ_LEN` so you can sweep shapes without editing code.",
        "`flashmla_kernel.cu` includes the Blackwell-specific tensor memory guard to keep compilation healthy on SM121 hardware.",
    ],
)

ENTRIES["ch19"] = chapter_entry(
    slug="ch19",
    title="Chapter 19 - Low-Precision Training & Memory Systems",
    summary=dedent(
        """\
        Explores NVFP4/FP8 workflows, KV-cache quantization, memory double buffering, and adaptive allocators so low-precision experiments remain numerically safe while squeezing every byte of HBM."""
    ),
    goals=[
        "Benchmark FP4/FP6/FP8 training loops with calibration and validation hooks.",
        "Overlap KV-cache prefetch with compute while respecting precision constraints.",
        "Implement dynamic quantized caches that switch formats mid-run without drift.",
        "Design allocator helpers to monitor and rebalance fragmented memory pools.",
    ],
    contents=[
        ("`baseline_nvfp4_training.py`, `optimized_nvfp4_training.py`, `native_fp4_quantization.py`, `native_fp6_quantization.py`, `native_fp8_training.py`", "Training and quantization recipes that switch between FP8 and NVFP4 with automatic calibration."),
        ("`baseline_memory_double_buffering.py`, `optimized_memory_double_buffering.py`, `memory_allocator_with_monitoring.py`, `dynamic_memory_allocator.py`, `_allocator_worker.py`", "Memory-management helpers covering double buffering, instrumentation, and adaptive worker pools."),
        ("`baseline_kv_prefetch_overlap.cu`, `optimized_kv_prefetch_overlap.cu`, `kv_prefetch_overlap_sm121` binaries", "CUDA kernels proving that quantized KV prefetch can overlap with compute when using cp.async pipelines."),
        ("`baseline_dynamic_quantized_cache.py`, `optimized_dynamic_quantized_cache.py`, `dynamic_quantized_cache.py`, `token_precision_switching.py`, `dynamic_precision_switching.py`", "Quantized cache management for dynamically switching between precisions based on accuracy budgets."),
        ("`fp4_hardware_kernel.cu`, `fp8_hardware_kernel.cu`, `custom_allocator_retry.py`, `adaptive_parallelism_strategy.py`, `adaptive_parallelism_worker_pool.py`", "Hardware-level kernels and adaptive scheduling helpers for heterogeneous precision fleets."),
        ("`compare.py`, `arch_config.py`, `expectations_gb10.json`", "Harness entry, architecture toggles, and stored expectation data."),
    ],
    validation=[
        "`python optimized_nvfp4_training.py --calibrate` warms up with FP8, then switches to NVFP4 and matches the baseline's accuracy thresholds.",
        "`python optimized_dynamic_quantized_cache.py --trace` logs precision transitions with bounded error, confirming correctness of token-level switching.",
        "`nvcc -o optimized_kv_prefetch_overlap_sm121 optimized_kv_prefetch_overlap.cu` plus the baseline binary show measurable overlap improvements in Nsight Compute.",
    ],
    notes=[
        "`arch_config.py` exposes `ENABLE_NVFP4`/`ENABLE_TF32` toggles per device, making it easy to compare precision recipes.",
        "`validate_quantization_performance.py` aggregates accuracy vs throughput numbers into CSV form for proof-of-benefit reporting.",
    ],
)

ENTRIES["ch20"] = chapter_entry(
    slug="ch20",
    title="Chapter 20 - End-to-End Case Studies",
    summary=dedent(
        """\
        Combines kernel, memory, pipeline, and inference optimizations into holistic case studies: take a baseline pipeline, apply staged improvements, and capture proof-of-benefit artifacts for every major subsystem."""
    ),
    goals=[
        "Chain memory, pipeline, and KV-cache optimizations together to see cumulative impact.",
        "Generate automatic reports that compare baseline vs tuned end-to-end runs.",
        "Prototype new kernels via the AI kernel generator and slot them into the harness.",
        "Validate improvements with workload-specific acceptance tests.",
    ],
    contents=[
        ("`baseline_multiple_unoptimized.py`, `optimized_multiple_unoptimized.py`, `ai_kernel_generator.py`, `inductor_guard.py`", "Composite workloads that stack several bottlenecks plus helpers for generating candidate kernels safely."),
        ("`baseline_pipeline_sequential.py`, `optimized_pipeline_sequential.py`, `baseline_end_to_end_bandwidth.py`, `optimized_end_to_end_bandwidth.py`", "Pipeline and bandwidth case studies showing how optimizations interact across stages."),
        ("`baseline_integrated_kv_cache.py`, `optimized_integrated_kv_cache.py`", "Integrated KV-cache demos that merge allocator, overlap, and NVLink pooling tricks."),
        ("`baseline_memory_standard.py`, `optimized_memory_standard.py`", "Memory-focused harness verifying allocator changes at system level."),
        ("`baseline_training_single.py`, `optimized_training_single.py`, `test.cu`, `Makefile`", "Single-device training case study plus CUDA kernels used in the final report."),
        ("`compare.py`, `arch_config.py`, `expectations_gb10.json`", "Harness driver, architecture settings, and expectation baselines."),
    ],
    validation=[
        "`python compare.py` emits per-stage summaries that show each optimized variant meeting or exceeding stored expectations.",
        "`python ai_kernel_generator.py --emit test.cu` produces CUDA kernels that compile via `nvcc` and integrate into the harness without manual edits.",
        "`python optimized_pipeline_sequential.py --trace` shows smooth NVTX ranges covering the entire pipeline, demonstrating overlap success.",
    ],
    notes=[
        "`inductor_guard.py` provides convenience toggles for gating experimental kernels behind feature flags.",
        "`ai_kernel_generator.py` logs generated code to `artifacts/` for reproducibility; capture the log with your proof-of-benefit bundle.",
    ],
)

ENTRIES["labs/blackwell_matmul"] = lab_entry(
    slug="labs/blackwell_matmul",
    title="Lab - Blackwell Matmul Suite",
    summary=dedent(
        """\
        Ports the four-part Blackwell matmul deep dive into the harness: start with a naive CUDA kernel, then layer pipeline loads, real TMA, and cluster DSMEM broadcasts until you surpass the baseline roofline."""
    ),
    goals=[
        "Reproduce the reference matmul trajectory (baseline -> pipelined -> TMA -> cluster).",
        "Compare PyTorch harness timings against the CUDA extensions while reusing the same shapes.",
        "Validate kernels on SM100/103 targets and gracefully skip DSMEM-only paths on SM121.",
        "Capture dual roofline metadata (SM vs TMEM) for every variant.",
    ],
    contents=[
        ("`baseline_blackwell_matmul.py`, `optimized_blackwell_matmul_pipeline.py`, `optimized_blackwell_matmul_tma.py`, `optimized_blackwell_matmul_cluster.py`", "Python entrypoints for each stage of the matmul tutorial."),
        ("`blackwell_benchmarks.py`, `run_blackwell_matmul.py`", "Harness adapters and standalone runner for quick sweeps and metadata capture."),
        ("`grace_blackwell_extension.py`, `grace_blackwell_kernels.cu`", "PyTorch extension and CUDA kernels implementing the baseline and optimized kernels."),
    ],
    validation=[
        "`python tools/cli/aisp bench run --targets labs/blackwell_matmul:blackwell_matmul_cluster --profile minimal` delivers higher TFLOP/s than the baseline and emits artifacts under `artifacts/labs_blackwell_matmul*`.",
        "`python labs/blackwell_matmul/run_blackwell_matmul.py --variant pipeline --size 4096 --roofline-meta artifacts/matmul_meta.csv` saves roofline metadata alongside timings.",
        "DSM-aware variants error out early on GPUs that lack cluster DSMEM support, preventing misleading results.",
    ],
    notes=[
        "`run_blackwell_matmul.py` accepts `--variant baseline|pipeline|tma|cluster` plus `--size` to mirror the blog walkthrough.",
        "TMA kernels require CUDA 13.0+ and SM100/103 hardware; on GB10 they log a warning and skip execution.",
    ],
)

ENTRIES["labs/cutlass_profiler_kernel_selector"] = lab_entry(
    slug="labs/cutlass_profiler_kernel_selector",
    title="Lab - CUTLASS Profiler Kernel Selector",
    summary=dedent(
        """\
        Automates CUTLASS profiler sweeps for transformer-style GEMMs, records Triton or custom kernel results, and compares everything so you can prove custom kernels beat the fastest stock CUTLASS option."""
    ),
    goals=[
        "Generate per-shape CUTLASS profiler logs and store the best kernel metadata.",
        "Optionally benchmark Triton or custom paths on the same shapes.",
        "Compare providers (CUTLASS, Triton, DeepEP, custom) with a uniform JSON schema.",
        "Adjust shapes quickly by editing a single definition file.",
    ],
    contents=[
        ("`run_cutlass_profiler_sweep.py`", "Invokes `cutlass_profiler` for every shape in `shapes.py` and stores JSON summaries."),
        ("`run_triton_matmul_baseline.py`", "Optional Triton matmul baseline for parity checks."),
        ("`compare_against_baselines.py`", "Reads CUTLASS + competitor JSON files and emits TFLOP/s + speedup tables."),
        ("`shapes.py`", "Central list of GEMM shapes (prefill, decode, KV proj, etc.)."),
    ],
    run=RunSection(
        commands=[
            "cd ai-performance-engineering",
            "python labs/cutlass_profiler_kernel_selector/run_cutlass_profiler_sweep.py --output-dir artifacts/cutlass_profiler",
            "python labs/cutlass_profiler_kernel_selector/run_triton_matmul_baseline.py --output-dir artifacts/cutlass_profiler",
            "python labs/cutlass_profiler_kernel_selector/compare_against_baselines.py --include-default-triton",
        ],
        notes=[
            "Set `CUTLASS_PROFILER_BIN` to point at your `cutlass_profiler` binary after running `./setup.sh`.",
            "Add extra providers by writing JSON files matching the documented schema (see `compare_against_baselines.py`).",
        ],
    ),
    validation=[
        "Profiler runs emit `artifacts/cutlass_profiler/cutlass_profiler_results.json` with per-shape winners; rerun when upgrading CUDA or GPUs.",
        "Triton baselines land in `artifacts/cutlass_profiler/triton_matmul_results.json` and should stay within a few percent of CUTLASS for supported shapes.",
        "`compare_against_baselines.py` exits non-zero when provided result files are missing records, ensuring CI catches stale outputs.",
    ],
    notes=[
        "Shapes can be overridden via CLI flags (e.g., `--shapes decode_mlp_m4096_n4096_k8192`).",
        "Provider JSON files may include metadata (kernel names, launch params) for additional debugging.",
    ],
)

ENTRIES["labs/cudnn_sdpa_bench"] = lab_entry(
    slug="labs/cudnn_sdpa_bench",
    title="Lab - cuDNN SDPA Bench",
    summary=dedent(
        """\
        Microbenchmarks cuDNN fused scaled-dot-product attention against Flash and math backends with explicit CLI backend selection."""
    ),
    goals=[
        "Compare cuDNN fused SDPA to Flash and math backends on identical shapes.",
        "Capture Nsight traces per backend to inspect kernel fusion and launch counts.",
        "Keep regression thresholds per architecture in `expectations_gb10.json`.",
    ],
    contents=[
        ("`baseline_flash_sdp.py`, `optimized_flash_sdp.py`", "Shared attention microbenchmarks; backend chosen via `--backend {auto,cudnn,flash,math}` passed with `--target-extra-arg`."),
        ("`expectations_gb10.json`", "Current golden timings on GB10 for regression checking."),
    ],
    validation=[
        "`python tools/cli/aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --profile minimal --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp=\"--backend cudnn\"` captures cuDNN with Nsight traces.",
        "`python tools/cli/aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp=\"--backend flash\"` compares the Flash path against cuDNN.",
        "`python tools/cli/aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp=\"--backend math\"` sanity-checks the math backend where fused kernels are unsupported.",
    ],
    notes=[
        "Backend selection is CLI-only; environment variables are intentionally ignored.",
        "Profiling outputs are stored under `benchmark_profiles/labs/cudnn_sdpa_bench/<run_id>` with harness artifacts in `artifacts/<run_id>/`.",
    ],
)

ENTRIES["labs/dynamic_router"] = lab_entry(
    slug="labs/dynamic_router",
    title="Lab - Dynamic Prefill/Decode Router",
    summary=dedent(
        """\
        Simulates and benchmarks dynamic routing policies for large-scale inference: split GPUs into prefill/decode pools, monitor TTFT/TPOT, honor KV locality, and migrate traffic only when the score gap warrants it."""
    ),
    goals=[
        "Compare naive round-robin routing with telemetry-driven policies that stabilize TTFT.",
        "Prototype migration budgets, KV-locality boosts, and per-pool thresholds.",
        "Drive the router against synthetic workloads or real vLLM engines.",
        "Export detailed metrics (TTFT, TPOT, queue depth) for visualization.",
    ],
    contents=[
        ("`baseline_router.py`, `optimized_router.py`, `driver.py`", "Core router logic plus a synthetic simulator for deterministic comparisons."),
        ("`baseline_dynamic_router.py`, `optimized_dynamic_router.py`", "Harness-facing benchmarks derived from the simulator."),
        ("`baseline_dynamic_router_vllm.py`, `optimized_dynamic_router_vllm.py`, `vllm_runner.py`", "Integrations for running the routing policy against vLLM instances."),
        ("`baseline_dual_pool_vllm.py`, `optimized_dual_pool_vllm.py`", "Shared-pool vs dual-pool TTFT benchmarks that reuse `vllm_runner.py`."),
        ("`topology.py`, `topology_probe.py`", "NUMA-aware GPU mapping helpers and a target that emits `artifacts/topology/topology.json` for routing hints."),
    ],
    validation=[
        "`python labs/dynamic_router/driver.py --mode baseline` vs `--mode optimized` shows lower TTFT variance and higher TPOT for the optimized policy.",
        "`python tools/cli/aisp bench run --targets labs/dynamic_router --profile minimal` records artifacts comparing baseline/optimized harness runs.",
        "`python tools/cli/aisp bench run --targets labs/dynamic_router:dynamic_router_vllm --target-extra-arg labs/dynamic_router:dynamic_router_vllm=\"--model /path/to/model --decode-gpus 0,1\"` succeeds on hosts with at least two GPUs and a local model copy.",
        "`python tools/cli/aisp bench run --targets labs/dynamic_router:dual_pool_vllm --target-extra-arg labs/dynamic_router:dual_pool_vllm=\"--model /path/to/model --prefill-gpus 0 --decode-gpus 1\"` contrasts shared versus dual pools and emits per-pool TTFT and queue depth.",
        "`python tools/cli/aisp bench run --targets labs/dynamic_router:topology_probe` captures GPU↔NUMA mappings and distance matrices for consumption by the router.",
    ],
    notes=[
        "`driver.py` accepts knobs such as `--prefill-gpus`, `--decode-gpus`, and `--migration-budget` to stress different regimes.",
        "vLLM integration now takes flags (`--model`, `--prefill-gpus`, `--decode-gpus`, etc.) plus locally available tokenizer/model weights.",
        "Router scoring incorporates pinned-host KV slab availability and NUMA-locality bias; feed it real topology via `topology_probe.py` or NVML when available.",
    ],
)

ENTRIES["labs/flexattention"] = lab_entry(
    slug="labs/flexattention",
    title="Lab - FlexAttention Harness",
    summary=dedent(
        """\
        Mirrors the FlexAttention CuTe DSL walkthrough: run eager vs compiled FlexAttention, compare to the CuTe path, and experiment with block masks, score modifiers, and Triton-style compilation."""
    ),
    goals=[
        "Benchmark FlexAttention eager mode against compiled variants using identical masks/score mods.",
        "Validate CuTe-based FlashAttention fallbacks for platforms where FlexAttention is not available.",
        "Sweep sparsity knobs (block size, doc span) without editing source.",
        "Collect Nsight traces showing kernel fusion improvements after compiling.",
    ],
    contents=[
        ("`baseline_flex_attention.py`, `optimized_flex_attention.py`", "FlexAttention DSL workloads toggling `torch.compile` for fused kernels."),
        ("`baseline_flex_attention_cute.py`, `optimized_flex_attention_cute.py`", "CuTe/FlashAttention versions for hardware without FlexAttention support."),
        ("`flexattention_common.py`, `expectations_gb10.json`", "Shared input builders, score modifiers, and regression thresholds."),
    ],
    validation=[
        "`python tools/cli/aisp bench run --targets labs/flexattention:flex_attention --profile minimal` captures the eager vs compiled delta and stores artifacts.",
        "`BLOCK_SIZE=64 DOC_SPAN=128 python tools/cli/aisp bench run --targets labs/flexattention:flex_attention` demonstrates masked sparsity sweeps.",
        "`python tools/cli/aisp bench run --targets labs/flexattention:flex_attention_cute` succeeds even on systems missing FlexAttention bindings.",
    ],
    notes=[
        "Environment variables such as `BLOCK_SIZE`, `DOC_SPAN`, and `TORCH_COMPILE_MODE` are read at runtime for quick experiments.",
        "Artifacts include NVTX traces; feed them to `tools/analysis/deep_profiling_report.py` for convenience.",
    ],
)

ENTRIES["labs/fullstack_cluster"] = lab_entry(
    slug="labs/fullstack_cluster",
    title="Lab - Full-Stack Blackwell Cluster",
    summary=dedent(
        """\
        Replays the entire performance-engineering arc as scenarios: from system prep to streaming inference, plus the original cluster GEMM CUDA kernels wired into the harness."""
    ),
    goals=[
        "Run scenario benchmarks that stitch together chapters into end-to-end workflows.",
        "Inspect cluster GEMM kernels (baseline and DSMEM/TMA optimized) via the CUDA extension.",
        "Track GPU requirements, expected shapes, and automation scripts in one place.",
        "Collect artifact bundles that summarize every phase of the scenario.",
    ],
    contents=[
        ("`baseline_01_system_foundations.py` ... `baseline_09_end_to_end.py`, `optimized_01_system_foundations.py` ... `optimized_09_end_to_end.py`, `scenario_benchmark.py`", "Scenario scripts that orchestrate system, kernel, compiler, memory, serving, and end-to-end phases."),
        ("`baseline_cluster_gemm.py`, `optimized_cluster_gemm.py`, `baseline_cluster_gemm_tcgen05.py`, `optimized_cluster_gemm_tcgen05.py`", "Python entrypoints for the cluster GEMM kernels with tcgen05 fallbacks."),
        ("`capstone_extension.py`, `capstone_kernels.cu`, `capstone_kernels_tcgen05.cu`, `capstone_benchmarks.py`", "PyTorch extension, CUDA kernels, and harness hooks for the GEMM showcase."),
        ("`run_lab_fullstack_cluster.py`, `gpu_requirements.py`, `expectations_gb10.json`", "Standalone runner, hardware requirement helper, and expectation file."),
    ],
    validation=[
        "`python tools/cli/aisp bench run --targets labs/fullstack_cluster --profile minimal` records per-phase metrics for the entire scenario suite.",
        "`python labs/fullstack_cluster/run_lab_fullstack_cluster.py --size 2048` builds the extension on first run and prints baseline vs optimized TFLOP/s.",
        "KF-specific kernels skip gracefully on hardware lacking tcgen05 or DSMEM, ensuring CI signal stays meaningful.",
    ],
    notes=[
        "`gpu_requirements.py` reports the minimum GPU count, memory, and features for each scenario; consult it before scheduling runs.",
        "`capstone_extension.py` caches builds under `~/.cache/torch_extensions`; run `python cleanup.py --include-extensions` when switching CUDA versions.",
    ],
)

ENTRIES["labs/moe_cuda"] = lab_entry(
    slug="labs/moe_cuda",
    title="Lab - CUDA MoE Decode Toolkit",
    summary=dedent(
        """\
        Implements mixture-of-experts decode helpers directly in CUDA: decode kernels, KV-transfer graphs, router policies, and validation math so you can iterate on Blackwell-friendly pipelines."""
    ),
    goals=[
        "Benchmark decode kernels that stage tokens through shared memory and cp.async pipelines.",
        "Optimize KV-transfer strategies (manual, CUDA Graphs) across NVLink fabrics.",
        "Prototype routers that understand MoE grouping, locality, and vectorized loads.",
        "Validate CUDA kernels against Python math models before integrating into serving stacks.",
    ],
    contents=[
        ("`baseline_decode_attention.py`, `optimized_decode_attention.py`, `optimized_decode_attention_math.py`", "Attention microbenchmarks plus math-only checks to vet numerical stability."),
        ("`baseline_decode_kernel.py`, `optimized_decode_kernel.py`, `decode_kernels.cu`, `kernels/`", "CUDA kernels and wrappers for the decode core."),
        ("`baseline_kv_transfer.py`, `optimized_kv_transfer.py`, `optimized_kv_transfer_graphs.py`", "KV-transfer samples comparing eager vs CUDA Graph orchestration."),
        ("`baseline_router.py`, `optimized_router.py`, `optimized_router_vectorized.py`", "MoE router logic fit for device execution."),
        ("`expectations_gb10.json`, `__init__.py`", "Metadata and module exports needed by the harness."),
    ],
    validation=[
        "`python tools/cli/aisp bench run --targets labs/moe_cuda --profile minimal` runs every baseline/optimized pair and captures NVTX traces.",
        "`python labs/moe_cuda/optimized_decode_attention_math.py --validate` compares the CUDA path to the math reference and fails loudly if drift is detected.",
        "KV transfer graphs print latency breakdowns showing overlap improvements relative to the baseline script.",
    ],
    notes=[
        "`kernels/` houses the raw CUDA sources split by component; edit schedules there before rebuilding via the harness.",
        "`optimized_kv_transfer_graphs.py` emits CUDA Graph captures under `artifacts/` for reproducibility.",
    ],
)

ENTRIES["labs/moe_parallelism"] = lab_entry(
    slug="labs/moe_parallelism",
    title="Lab - MoE Parallelism Planner",
    summary=dedent(
        """\
        Provides scenario planning for mixture-of-experts clusters: memory budgeting, network affinity, parallelism breakdown, and pipeline schedules expressed as baseline/optimized harness targets."""
    ),
    goals=[
        "Quantify memory budgets for experts, routers, and KV caches before deploying models.",
        "Explore different grouping strategies (hashing, topology-aware) and their throughput impact.",
        "Model network affinity to decide where experts should live in an NVLink/NVSwitch fabric.",
        "Simulate pipeline schedules to identify bottlenecks before touching production systems.",
    ],
    contents=[
        ("`baseline_memory_budget.py`, `optimized_memory_budget.py`", "Memory planners that prove how optimized layouts free additional HBM for experts."),
        ("`baseline_moe_grouping.py`, `optimized_moe_grouping.py`, `plan.py`", "Grouping strategies spanning naive, locality-aware, and latency-balanced heuristics."),
        ("`baseline_network_affinity.py`, `optimized_network_affinity.py`", "Network-affinity calculators comparing NVLink, NVSwitch, and PCIe hops."),
        ("`baseline_parallelism_breakdown.py`, `optimized_parallelism_breakdown.py`, `baseline_pipeline_schedule.py`, `optimized_pipeline_schedule.py`", "Parallelism and scheduling studies for sharding experts across GPUs."),
        ("`benchmarking.py`, `run_lab.py`, `__init__.py`", "Lab driver, Typer CLI, and exports used by the harness."),
    ],
    validation=[
        "`python tools/cli/aisp bench run --targets labs/moe_parallelism --profile minimal` runs every planner pair and drops JSON/Markdown summaries.",
        "`python labs/moe_parallelism/run_lab.py --scenario grouped` prints an actionable plan (experts/GPU, bandwidth needs) for the chosen scenario.",
        "`python labs/moe_parallelism/optimized_memory_budget.py --validate` ensures optimized allocations meet the same correctness checks as the baseline.",
    ],
    notes=[
        "`plan.py` centralizes scenario definitions so you only update one file when adding a new MoE topology.",
        "`benchmarking.py` can emit Markdown tables for documentation by passing `--format markdown`.",
    ],
)

ENTRIES["labs/occupancy_tuning"] = lab_entry(
    slug="labs/occupancy_tuning",
    title="Lab - Triton Occupancy & Schedule Sweep",
    summary=dedent(
        """\
        Sweeps Triton matmul schedules for ProtonNet-style workloads on Blackwell, comparing the baseline schedule against optimized block/warp dimensions and reporting how each choice affects occupancy and FLOP/s."""
    ),
    goals=[
        "Measure how Triton block sizes map to achieved occupancy on SM100/121.",
        "Autogenerate schedule sweeps and record best-performing parameter sets.",
        "Compare baseline schedules to curated optimized variants packaged with the lab.",
        "Integrate selected schedules into harness targets for regression tracking.",
    ],
    contents=[
        ("`baseline_proton_matmul.py`, `optimized_proton_matmul_bm128_bn128_bk32_nw8.py`, `optimized_proton_matmul_bm64_bn64_bk32_nw2.py`, `optimized_proton_matmul_bm64_bn256_bk32.py`, `optimized_proton_matmul_bm128_bn256_bk64.py`", "Baseline and optimized Triton schedules covering multiple block/warp configurations."),
        ("`triton_matmul.py`, `triton_matmul_schedules.py`", "Core Triton kernel and schedule definitions used by the harness."),
        ("`sweep_schedules.py`", "Utility for enumerating candidate schedules and logging throughput/occupancy to `artifacts/`."),
    ],
    validation=[
        "`python tools/cli/aisp bench run --targets labs/occupancy_tuning --profile minimal` executes every schedule defined in the lab.",
        "`python labs/occupancy_tuning/sweep_schedules.py --output artifacts/occupancy_tuning.csv` enumerates schedules and highlights the top performer.",
        "`python labs/occupancy_tuning/optimized_proton_matmul_bm128_bn128_bk32_nw8.py --validate` compares outputs against the baseline to ensure correctness.",
    ],
    notes=[
        "Add new schedules to `triton_matmul_schedules.py` and regenerate the harness targets by rerunning the sweep script.",
        "`expectations_gb10.json` records FLOP/s per schedule so improvements show up in CI.",
    ],
)

ENTRIES["labs/persistent_decode"] = lab_entry(
    slug="labs/persistent_decode",
    title="Lab - Persistent Decode & TMA Prefill",
    summary=dedent(
        """\
        Demonstrates Blackwell-friendly persistent decode kernels and TMA-powered prefill paths, all validated via Python harnesses plus CUDA/Triton implementations."""
    ),
    goals=[
        "Contrast naive decode loops against persistent kernels that pin CTAs per sequence.",
        "Adopt TMA-based prefill to stream activations into shared memory with minimal latency.",
        "Benchmark CUDA vs Triton implementations with unified validation utilities.",
        "Mix CUDA Graphs into the decode path to remove residual launch overhead.",
    ],
    contents=[
        ("`baseline_persistent_decode.py`, `optimized_persistent_decode_cuda.py`, `optimized_persistent_decode_graphs.py`, `optimized_persistent_decode_triton.py`", "Persistent decode variants spanning CUDA, graphs, and Triton."),
        ("`baseline_tma_prefill_decode.py`, `optimized_tma_prefill_decode.py`, `baseline_native_tma_prefill_decode.py`, `optimized_native_tma_prefill_decode.py`", "Prefill workloads illustrating cp.async vs native TMA scheduling."),
        ("`kv_locality_microbench.py`", "Pinned/pageable/NUMA host slab copy microbench (HBM vs local/remote pinned vs pageable)."),
        ("`persistent_decode_common.py`, `tma_extension.py`, `expectations_gb10.json`", "Shared helpers, CUDA extension wrappers, and expectation thresholds."),
    ],
    validation=[
        "`python tools/cli/aisp bench run --targets labs/persistent_decode --profile minimal` compares all persistent/TMA variants in one sweep.",
        "`python labs/persistent_decode/optimized_persistent_decode_graphs.py --iterations 50` shows lower launch overhead than `baseline_persistent_decode.py`.",
        "`python labs/persistent_decode/optimized_native_tma_prefill_decode.py --validate` matches the math reference while reporting achieved memory throughput.",
        "`python labs/persistent_decode/kv_locality_microbench.py` surfaces H2D copy time deltas for pageable vs pinned slabs; add `QUICK=1` for a short smoke run.",
    ],
    notes=[
        "Set `TORCH_COMPILE_MODE` or `TMA_TILE_SIZE` via env vars before invoking the harness to sweep tile sizes.",
        "`tma_extension.py` caches builds under `~/.cache/torch_extensions`; clean the cache when switching CUDA versions.",
    ],
)

ENTRIES["labs/train_distributed"] = lab_entry(
    slug="labs/train_distributed",
    title="Lab - Distributed Training Playbook",
    summary=dedent(
        """\
        Collects distributed-training recipes for Blackwell clusters: DDP, FSDP, ZeRO-1/2/3, symmetric memory, and flash-attention-aware all-reduce handling, all runnable through the harness."""
    ),
    goals=[
        "Benchmark standard DDP vs optimized overlap-aware variants.",
        "Exercise FSDP and ZeRO strategies with shared helper utilities.",
        "Validate symmetric-memory training modes that pool NVLink bandwidth.",
        "Reuse launcher utilities (torchrun) with consistent configuration.",
    ],
    contents=[
        ("`baseline_ddp.py`, `optimized_ddp.py`, `baseline_ddp_flash.py`, `optimized_ddp_flash.py`, `ddp.py`", "DDP workloads including flash-attention aware overlap tuning."),
        ("`baseline_fsdp.py`, `optimized_fsdp.py`, `train_fsdp.py`", "FSDP scripts that demonstrate shard-by-shard memory savings."),
        ("`baseline_symmem_training.py`, `optimized_symmem_training.py`", "Symmetric-memory strategies for optimizer state replication."),
        ("`baseline_zero1.py`, `baseline_zero2.py`, `baseline_zero3.py`, `optimized_zero1.py`, `optimized_zero2.py`, `optimized_zero3.py`, `zero1.py`, `zero2.py`, `zero3.py`", "ZeRO implementations (1/2/3) plus helpers for parameter partitioning."),
        ("`training_utils/`, `utils.py`, `__init__.py`", "Shared launch utilities, argument parsing, and harness exports."),
    ],
    validation=[
        "`python tools/cli/aisp bench run --targets labs/train_distributed --profile minimal` runs every distributed configuration registered with the harness.",
        "`python labs/train_distributed/train_fsdp.py --validate` confirms numerical parity between FSDP shards and the baseline DDP path.",
        "`python labs/train_distributed/optimized_zero3.py --summary` shows reduced peak memory vs the baseline script.",
    ],
    notes=[
        "Set `TORCHRUN_ARGS` or pass `--torchrun-env` via the CLI when launching multi-node tests.",
        "`utils.py` exposes helper functions (like `resolve_topology()`) that can be reused in other labs.",
    ],
)

def main() -> None:
    for slug, entry in ENTRIES.items():
        if slug.endswith(".md"):
            output_path = REPO_ROOT / slug
        else:
            output_path = REPO_ROOT / slug / "README.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        markdown = _format_markdown(entry).rstrip() + "\n"
        output_path.write_text(markdown)
        print(f"Wrote {output_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
